use core::cmp::Ordering;
use core::convert::identity;
use core::fmt::Formatter;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use std::ops::{FromResidual, Try};

use libc::{pipe, wchar_t};
use libm::log;

use Condition::*;

use crate::gfx::vk::include::Instance;
use crate::gfx::Renderer;
use crate::prelude::*;

pub trait WorldState {
    fn spawn(&mut self) -> Entity;
    fn query<T: Fetch, F: Filter>(&mut self) -> Query<T, F>;
    fn insert(
        &mut self,
        entity: crate::ecs::Entity,
        components: impl crate::ecs::IntoComponents,
    ) -> crate::ecs::Entity;

    fn push(&mut self, components: impl crate::ecs::IntoComponents) -> crate::ecs::Entity {
        let entity = self.spawn();
        self.insert(entity, components)
    }
    fn extend<'a>(
        &'a mut self,
        components: impl IntoIterator<Item = impl crate::ecs::IntoComponents + 'a> + 'a,
    ) -> impl Iterator<Item = crate::ecs::Entity> + 'a {
        components.into_iter().map(|x| self.push(x))
    }
}

#[derive(Default, Debug)]
pub struct EntityPartition {
    entity_cursor: usize,
    entity_max: usize,
    entity_freed: Vec<Entity>,
}
impl EntityPartition {
    fn spawn(&mut self) -> Option<Entity> {
        if self.full() {
            None?
        }
        let entity = Entity {
            identifier: self.entity_cursor,
            generation: 0,
        };
        self.entity_cursor += 1;
        Some(entity)
    }
    fn full(&self) -> bool {
        self.entity_cursor >= self.entity_max - 1
    }
}
#[derive(Debug, Default)]
pub struct Entities {
    partitions: Vec<Option<EntityPartition>>,
}

impl Entities {
    pub(crate) fn put(&mut self, partition: EntityPartition) {
        let empty = self
            .partitions
            .iter()
            .enumerate()
            .find_map(|(i, x)| x.is_none().then_some(i))
            .unwrap();
        self.partitions[empty] = Some(partition);
    }
    fn new(partitions: usize) -> Self {
        let size = usize::MAX / partitions;
        let mut sizes = vec![];
        let mut cursor = 0;
        for i in 0..partitions {
            sizes.push(cursor);
            cursor += size;
        }

        let partitions = sizes
            .into_iter()
            .map(|entity_cursor| {
                Some(EntityPartition {
                    entity_cursor,
                    ..default()
                })
            })
            .collect();

        Self { partitions }
    }
    fn take(&mut self) -> EntityPartition {
        let avail = self
            .partitions
            .iter()
            .enumerate()
            .find_map(|(i, x)| x.is_some().then_some(i))
            .unwrap();
        let mut partition = None;
        mem::swap(&mut self.partitions[avail], &mut partition);
        partition.unwrap()
    }
    fn spawn(&mut self) -> Entity {
        self.partitions
            .iter_mut()
            .find_map(|x| try { x.as_mut()?.spawn() })
            .flatten()
            .unwrap()
    }
}

#[derive(Debug, Default)]
pub struct Columns {
    archetypes: BTreeSet<Archetype>,
    data: Vec<Option<Column>>,
}

impl Columns {
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }
}

impl Columns {
    pub(crate) fn get_mut(&mut self, archetype: Archetype) -> &mut Column {
        let index = self
            .archetypes
            .iter()
            .enumerate()
            .find_map(|(i, x)| (x == &archetype).then_some(i))
            .unwrap();
        self.data[index].as_mut().unwrap()
    }
}

impl Columns {
    pub(crate) fn insert(
        &mut self,
        archetype: Archetype,
        column: Option<Column>,
    ) -> Option<Column> {
        let ret = if self.archetypes.contains(&archetype) {
            let index = self
                .archetypes
                .iter()
                .enumerate()
                .find_map(|(i, x)| (x == &archetype).then_some(i))
                .unwrap();
            Some(self.data.remove(index))
        } else {
            None
        };
        self.archetypes.insert(archetype.clone());
        let index = self
            .archetypes
            .iter()
            .enumerate()
            .find_map(|(i, x)| (x == &archetype).then_some(i))
            .unwrap();
        self.data.insert(index, column);
        ret?
    }
}

impl Columns {
    fn subset(prototype: &Prototype) -> Self {
        todo!()
    }
}

impl Index<usize> for Columns {
    type Output = Column;

    fn index(&self, index: usize) -> &Self::Output {
        self.data[index].as_ref().unwrap()
    }
}

impl IndexMut<usize> for Columns {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.data[index].as_mut().unwrap()
    }
}

#[derive(Debug, Default)]
pub struct World {
    entities: Entities,
    columns: Columns,
}

#[derive(Clone, Copy)]
pub enum TakenMeta {
    InlineQuery {
        src_file: &'static str,
        src_line: usize,
    },
}

impl World {
    pub(crate) fn with_partitions(num: usize) -> Self {
        Self {
            columns: default(),
            entities: Entities::new(num),
        }
    }
    pub(crate) fn take_partition(&mut self) -> EntityPartition {
        self.entities.take()
    }
    pub(crate) fn consume_state(&mut self, state: World) {
        for (archetype, column) in state.columns.archetypes.into_iter().zip(state.columns.data) {
            self.columns.insert(archetype, column);
        }
        for partition in state.entities.partitions {
            if let Some(p) = partition {
                self.entities.put(p);
            }
        }
    }
    pub(crate) fn sub_state(&mut self, archetype_query: &Archetype) -> World {
        let mut take = vec![];

        for archetype_column in &self.columns.archetypes {
            if archetype_column.is_superset(&archetype_query) {
                take.push(archetype_column.clone());
            }
        }

        let archetypes = take.into_iter().collect::<BTreeSet<_>>();
        let data = archetypes
            .iter()
            .map(|archetype| self.columns.insert(archetype.clone(), None))
            .collect::<Vec<_>>();

        World {
            columns: Columns { archetypes, data },
            entities: Entities {
                partitions: vec![Some(self.take_partition())],
            },
        }
    }
}

impl WorldState for World {
    fn spawn(&mut self) -> Entity {
        self.entities.spawn()
    }

    fn query<T: Fetch, F: Filter>(&mut self) -> Query<T, F> {
        let sub_world = self.sub_state(&T::archetype());
        Query {
            sub_world,
            components: PhantomData,
            filter: F::new(),
        }
    }
    fn insert(
        &mut self,
        entity: Entity,
        components: impl crate::ecs::IntoComponents,
    ) -> crate::ecs::Entity {
        let mut components = components.into_components();
        components.sort();
        let archetype = components
            .iter()
            .map(|(meta, _)| meta)
            .copied()
            .collect::<Archetype>();
        let data = components
            .into_iter()
            .flat_map(|(_, data)| data)
            .collect::<Vec<_>>();
        if !self.columns.archetypes.contains(&archetype) {
            self.columns.insert(
                archetype.clone(),
                Some(Column::with_archetype(archetype.clone())),
            );
        }
        let column = self.columns.get_mut(archetype.clone());

        let x = column.insert(Row {
            entity,
            data,
            archetype,
        });

        if let Some(x) = x {
            todo!()
        }

        entity
    }
}

#[derive(Debug, Ord, PartialOrd, Clone, Copy, Eq)]
pub struct Entity {
    identifier: usize,
    generation: usize,
}

impl PartialEq for Entity {
    fn eq(&self, other: &Self) -> bool {
        if self.identifier == other.identifier {
            assert_eq!(self.generation, other.generation);
            true
        } else {
            false
        }
    }
}

pub trait Component: 'static + Sized {
    fn identifier() -> ComponentId {
        ComponentId(TypeId::of::<Self>())
    }
    fn meta() -> ComponentMeta {
        ComponentMeta {
            identifier: Self::identifier(),
            access: None,
            name: type_name::<Self>(),
            size: mem::size_of::<Self>(),
        }
    }
}

pub trait IntoComponents {
    fn into_components(self) -> Vec<(ComponentMeta, Vec<u8>)>;
}

macro_rules! tuple_into_components {
    ($($arg:tt, $var:tt);*) => {
	    impl<$($arg: Component),*> IntoComponents for ($($arg,)*) {
		    fn into_components(self) -> Vec<(ComponentMeta, Vec<u8>)> {
			    let ($($var,)*) = self;
			    vec![$(($arg::meta(), erase($arg::identifier(), $var)),)*]
		    }
	    }
    };
}

tuple_into_components!(A, a);
tuple_into_components!(A, a; B, b);
tuple_into_components!(A, a; B, b; C, c);
tuple_into_components!(A, a; B, b; C, c; D, d);
tuple_into_components!(A, a; B, b; C, c; D, d; E, e);
tuple_into_components!(A, a; B, b; C, c; D, d; E, e; F, f);
tuple_into_components!(A, a; B, b; C, c; D, d; E, e; F, f; G, g);
tuple_into_components!(A, a; B, b; C, c; D, d; E, e; F, f; G, g; H, h);

#[derive(Eq, PartialEq, Clone, Debug, Ord)]
pub struct Archetype {
    meta: BTreeSet<ComponentMeta>,
    stride: Vec<usize>,
    size: usize,
}

impl Archetype {
    pub(crate) fn index_of(&self, p0: ComponentId) -> usize {
        self.meta
            .iter()
            .enumerate()
            .find_map(|(i, x)| (x.identifier == p0).then_some(i))
            .unwrap()
    }
    pub(crate) fn is_superset(&self, other: &Archetype) -> bool {
        let id_super = self
            .meta
            .iter()
            .map(|x| x.identifier)
            .collect::<BTreeSet<_>>();
        let id_sub = other
            .meta
            .iter()
            .map(|x| x.identifier)
            .collect::<BTreeSet<_>>();

        id_sub.difference(&id_super).count() == 0
    }
}

impl PartialOrd for Archetype {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.meta.partial_cmp(&other.meta)
    }
}

impl FromIterator<ComponentMeta> for Archetype {
    fn from_iter<T: IntoIterator<Item = ComponentMeta>>(iter: T) -> Self {
        let meta = iter.into_iter().collect::<BTreeSet<_>>();
        let stride = (0..meta.len())
            .map(|i| meta.iter().take(i).map(|x| x.size).sum())
            .collect::<Vec<_>>();
        let size = meta.iter().map(|x| x.size).sum();
        Self { stride, meta, size }
    }
}

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub struct ComponentId(TypeId);

#[derive(Clone, Copy, Debug, Ord, Eq, PartialEq)]
pub struct ComponentMeta {
    identifier: ComponentId,
    access: Option<Access>,
    name: &'static str,
    size: usize,
}

impl PartialOrd for ComponentMeta {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.identifier.partial_cmp(&other.identifier)
    }
}

pub struct Row {
    entity: Entity,
    data: Vec<u8>,
    archetype: Archetype,
}

#[derive(Debug)]
pub struct Column {
    entities: Vec<Entity>,
    data: Vec<u8>,
    stride: usize,
    archetype: Archetype,
}

impl Column {
    fn len(&self) -> usize {
        self.entities.len()
    }

    fn as_ptr(&mut self, index: usize) -> *mut u8 {
        unsafe { self.data.as_mut_ptr().add(index * self.stride) }
    }

    fn with_archetype(archetype: Archetype) -> Self {
        Self {
            entities: vec![],
            data: vec![],
            stride: archetype.size,
            archetype,
        }
    }

    fn insert(&mut self, row: Row) -> Option<Row> {
        if let old = self.remove(row.entity)
            && old.is_some()
        {
            return old;
        }

        assert_eq!(self.archetype, row.archetype);

        //assume the data is correct in the row
        self.data.extend(row.data);
        self.entities.push(row.entity);

        None
    }

    fn remove(&mut self, entity: Entity) -> Option<Row> {
        let idx = self.entities.binary_search(&entity).ok()?;
        let start = idx * self.stride;
        let data = self
            .data
            .splice(start..start + self.stride, iter::empty())
            .collect();
        Some(Row {
            entity,
            data,
            archetype: self.archetype.clone(),
        })
    }
}

pub trait Fetch {
    type Item;

    fn archetype() -> Archetype;
    fn local_stride(archetype: &Archetype) -> Vec<usize>;
    fn get(index: &mut usize, stride: &[usize], root: *mut u8) -> Self::Item;
}

impl<'a, T: Component> Fetch for &'a mut T {
    type Item = &'a mut T;

    fn archetype() -> Archetype {
        Archetype::from_iter([ComponentMeta {
            access: Some(Access::Mut),
            ..T::meta()
        }])
    }

    fn local_stride(archetype: &Archetype) -> Vec<usize> {
        vec![archetype.stride[archetype.index_of(T::identifier())]]
    }

    fn get(index: &mut usize, stride: &[usize], root: *mut u8) -> Self::Item {
        let item = interpret_mut(T::identifier(), stride[*index], root);
        *index += 1;
        item
    }
}

pub trait Filter {
    fn new() -> Self;
}

pub struct Passthrough;

impl Filter for Passthrough {
    fn new() -> Self {
        Self
    }
}

pub struct Query<T: Fetch, F: Filter = Passthrough> {
    sub_world: World,
    components: PhantomData<T>,
    filter: F,
}

impl<T: Fetch, F: Filter> Query<T, F> {
    pub fn single(self) -> T::Item {
        self.into_iter().next().unwrap()
    }
}

pub struct QueryIter<T: Fetch, F: Filter> {
    query: Query<T, F>,
    column_cursor: usize,
    row_cursor: usize,
    local_stride: Vec<Vec<usize>>,
}

impl<T: Fetch, F: Filter> IntoIterator for Query<T, F> {
    type Item = T::Item;
    type IntoIter = QueryIter<T, F>;

    fn into_iter(self) -> Self::IntoIter {
        let local_stride = self
            .sub_world
            .columns
            .archetypes
            .iter()
            .map(|a| T::local_stride(a))
            .collect();
        QueryIter {
            query: self,
            row_cursor: 0,
            column_cursor: 0,
            local_stride,
        }
    }
}

impl<T: Fetch, F: Filter> Iterator for QueryIter<T, F> {
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.column_cursor >= self.query.sub_world.columns.len() {
            None?
        }

        let column = &mut self.query.sub_world.columns[self.column_cursor];

        if self.row_cursor >= column.len() {
            self.row_cursor = 0;
            self.column_cursor += 1;
        }

        let root = column.as_ptr(self.row_cursor);
        let item = Some(T::get(&mut 0, &self.local_stride[self.column_cursor], root));

        self.row_cursor += 1;

        item
    }
}

pub fn erase<T: 'static>(id: impl ToId, t: T) -> Vec<u8> {
    assert_eq!(id.to_id(), TypeId::of::<T>());
    let t = ManuallyDrop::new(t);
    unsafe {
        ptr::slice_from_raw_parts(&t as *const _ as *const _, mem::size_of::<T>())
            .as_ref()
            .unwrap()
            .to_vec()
    }
}

pub fn interpret<'a, T: 'static>(id: impl ToId, offset: usize, v: *const u8) -> &'a T {
    assert_eq!(id.to_id(), TypeId::of::<T>());
    unsafe { (v.add(offset) as *const T).as_ref().unwrap() }
}

pub fn interpret_mut<'a, T: 'static>(id: impl ToId, offset: usize, v: *mut u8) -> &'a mut T {
    assert_eq!(id.to_id(), TypeId::of::<T>());
    unsafe { (v.add(offset) as *mut T).as_mut().unwrap() }
}

pub trait ToId {
    fn to_id(self) -> TypeId;
}

impl ToId for ComponentId {
    fn to_id(self) -> TypeId {
        self.0
    }
}

impl ToId for ResourceId {
    fn to_id(self) -> TypeId {
        self.0
    }
}

#[derive(Debug, Default)]
pub struct Resources {
    data: BTreeMap<ResourceId, Vec<u8>>,
}

impl Resources {
    pub(crate) fn get<T: Resource>(&self) -> Option<&T> {
        Some(interpret(
            T::identifier(),
            0,
            self.data.get(&T::identifier())?.as_ptr(),
        ))
    }
    pub fn insert<T: Resource>(&mut self, resource: T) {
        let data = erase(T::identifier(), resource);
        self.data.insert(T::identifier(), data);
    }
    pub(crate) fn consume_state(&mut self, p0: Resources) {
        for (k, v) in p0.data {
            self.data.insert(k, v);
        }
    }
    pub(crate) fn get_mut<T: Resource>(&mut self) -> Option<&mut T> {
        Some(interpret_mut(
            T::identifier(),
            0,
            self.data.get_mut(&T::identifier())?.as_mut_ptr(),
        ))
    }
    fn sub_state(&mut self, supertype: &Supertype) -> Self {
        let mut new = Self::default();
        for meta in &supertype.0 {
            let ty = meta.identifier;
            let name = meta.name;
            let Some(x) = self.data.remove(&ty) else {
                continue;
            };
            new.data.insert(ty, x);
        }
        new
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct ResourceId(TypeId);

pub trait Resource: 'static {
    fn identifier() -> ResourceId {
        ResourceId(TypeId::of::<Self>())
    }
}

pub struct BoxedSystem<Input, Output>(Box<dyn System<Input = Input, Output = Output>>);

impl<Input, Output> Deref for BoxedSystem<Input, Output> {
    type Target = Box<dyn System<Input = Input, Output = Output>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Input, Output> DerefMut for BoxedSystem<Input, Output> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(PartialOrd, Ord, Clone, Copy, Eq, PartialEq, Debug)]
pub struct SystemId(TypeId);

pub trait System {
    type Input;
    type Output;

    fn boxed(&self) -> BoxedSystem<Self::Input, Self::Output>;
    fn run(&mut self, input: Self::Input) -> Self::Output;

    fn id(&self) -> SystemId;
    fn name(&self) -> &str;
}

pub trait ToSystem<S: System> {
    fn to_system(&self) -> S;
    fn id(&self) -> SystemId;
    fn name(&self) -> &str;
}

pub trait Func<Input, Output> {
    fn boxed(&self) -> Box<dyn Func<Input, Output>>;
    fn execute(&mut self, input: Input) -> Output;
}

impl_system_func!();

pub struct FuncSystem<Input, Output> {
    func: Box<dyn Func<Input, Output>>,
    id: SystemId,
    name: String,
}

impl<Input: 'static, Output: 'static> System for FuncSystem<Input, Output> {
    type Input = Input;
    type Output = Output;

    fn boxed(&self) -> BoxedSystem<Self::Input, Self::Output> {
        let Self { func, name, id } = self;
        BoxedSystem(Box::new(Self {
            func: func.boxed(),
            name: name.clone(),
            id: id.clone(),
        }))
    }

    fn run(&mut self, input: Self::Input) -> Self::Output {
        self.func.execute(input)
    }

    fn id(&self) -> SystemId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl<F, Input: 'static, Output: 'static> ToSystem<FuncSystem<Input, Output>> for F
where
    F: Func<Input, Output>,
{
    fn to_system(&self) -> FuncSystem<Input, Output> {
        FuncSystem {
            func: self.boxed(),
            id: self.id(),
            name: self.name().to_string(),
        }
    }

    fn id(&self) -> SystemId {
        SystemId(typeid::of::<Self>())
    }

    fn name(&self) -> &'static str {
        type_name_of_val(self)
    }
}

#[derive(Clone, Ord, Eq, PartialEq)]
pub struct ResourceMeta {
    identifier: ResourceId,
    name: &'static str,
}

impl PartialOrd for ResourceMeta {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.identifier.partial_cmp(&other.identifier)
    }
}

pub struct Supertype(BTreeSet<ResourceMeta>);

impl FromIterator<ResourceMeta> for Supertype {
    fn from_iter<T: IntoIterator<Item = ResourceMeta>>(iter: T) -> Self {
        let ids = iter.into_iter().collect::<BTreeSet<_>>();
        Self(ids)
    }
}

impl From<Prototype> for Supertype {
    fn from(value: Prototype) -> Self {
        value
            .0
            .iter()
            .filter_map(|x| {
                let DataId::Resource(identifier) = x.data_id else {
                    None?
                };
                let name = x.name;
                Some(ResourceMeta { identifier, name })
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct Prototype(BTreeSet<TypeAccess>);

impl Extend<TypeAccess> for Prototype {
    fn extend<T: IntoIterator<Item = TypeAccess>>(&mut self, iter: T) {
        for i in iter {
            self.0.insert(i);
        }
    }
}

impl From<Archetype> for Prototype {
    fn from(value: Archetype) -> Self {
        Self(
            value
                .meta
                .iter()
                .map(|x| TypeAccess {
                    data_id: DataId::Component(x.identifier),
                    access: x.access.unwrap(),
                    name: x.name,
                    size: x.size,
                })
                .collect(),
        )
    }
}

impl From<Prototype> for Archetype {
    fn from(value: Prototype) -> Self {
        value
            .0
            .iter()
            .filter_map(|x| {
                let DataId::Component(identifier) = x.data_id else {
                    None?
                };
                Some(ComponentMeta {
                    identifier,
                    access: Some(x.access),
                    name: x.name,
                    size: x.size,
                })
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, Ord, Eq, PartialEq)]
pub enum Access {
    Ref,
    Mut,
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum DataId {
    Component(ComponentId),
    Resource(ResourceId),
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct TypeAccess {
    data_id: DataId,
    access: Access,
    name: &'static str,
    size: usize,
}

pub struct SetMarker<Marker, Other>(PhantomData<Marker>, PhantomData<Other>);
pub struct Set<Tuple>(pub Tuple);

impl_set_sequence!();

pub struct Before<A, B>(A, B);

pub struct After<A, B>(A, B);

pub struct Chain<Marker, Tuple>(Tuple, PhantomData<Marker>);

pub struct If<A, B>(A, B);
pub struct IfMarker<Marker, Other>(PhantomData<(Marker, Other)>);

pub trait Capture<Marker> {
    fn capture(&self, pipeline: &mut Pipeline);
    fn dependencies(&self) -> Vec<Dependency>;
}

pub trait Sequence<Marker>: Capture<Marker> + Sized {
    type Input;
    type Output;

    fn run_if<Other, Seq: Sequence<Other, Output = Condition>>(
        self,
        other: Seq,
    ) -> impl Sequence<IfMarker<Marker, Other>, Input = Self::Input, Output = Self::Output> {
        If(self, other)
    }

    fn after<Other, Seq>(
        self,
        other: Seq,
    ) -> impl Sequence<AfterMarker<Marker, Other>, Input = Self::Input, Output = Self::Output>
    where
        Seq: Sequence<Other>,
    {
        After(self, other)
    }

    fn before<Other, Seq>(
        self,
        other: Seq,
    ) -> impl Sequence<BeforeMarker<Marker, Other>, Input = Self::Input, Output = Self::Output>
    where
        Seq: Sequence<Other>,
    {
        Before(self, other)
    }

    fn chain(self) -> impl Sequence<SetMarker<Marker, Self>, Input = (), Output = ()> {
        Chain(self, PhantomData)
    }
}

impl<Marker, Tuple: Sequence<Marker>> Sequence<SetMarker<Marker, Tuple>> for Chain<Marker, Tuple> {
    type Input = ();
    type Output = ();
}

impl<Marker, Tuple: Sequence<Marker>> Capture<SetMarker<Marker, Tuple>> for Chain<Marker, Tuple> {
    fn capture(&self, pipeline: &mut Pipeline) {
        let Self(tuple, _) = self;
        let mut intra_pipeline = Pipeline::default();
        tuple.capture(&mut intra_pipeline);
        for (a, b) in intra_pipeline
            .steps
            .windows(2)
            .map(|x| (&x[0].job, &x[1].job))
        {
            pipeline
                .control_flow(b.id())
                .extend([ControlFlow::Depend(Dependency {
                    id: a.id(),
                    name: a.name().to_string(),
                })]);
        }
        tuple.capture(pipeline);
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let Self(tuple, _) = self;
        tuple.dependencies()
    }
}

impl<Marker, Other, A: Sequence<Marker>, B: Sequence<Other>> Sequence<IfMarker<Marker, Other>>
    for If<A, B>
{
    type Input = A::Input;
    type Output = A::Output;
}

impl<Marker, Other, A: Sequence<Marker>, B: Sequence<Other>> Capture<IfMarker<Marker, Other>>
    for If<A, B>
{
    fn capture(&self, pipeline: &mut Pipeline) {
        let Self(a, b) = self;

        let dependencies = b
            .dependencies()
            .into_iter()
            .map(ControlFlow::If)
            .collect::<Vec<_>>();

        for Dependency { id, .. } in a.dependencies() {
            pipeline.control_flow(id).extend(dependencies.clone());
        }

        a.capture(pipeline);
        b.capture(pipeline);
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let Self(a, _) = self;
        a.dependencies()
    }
}

pub struct BeforeMarker<Marker, Other>(PhantomData<(Marker, Other)>);

impl<Marker, Other, A: Sequence<Marker>, B: Sequence<Other>> Sequence<BeforeMarker<Marker, Other>>
    for Before<A, B>
{
    type Input = A::Input;
    type Output = A::Output;
}

impl<Marker, Other, A: Sequence<Marker>, B: Sequence<Other>> Capture<BeforeMarker<Marker, Other>>
    for Before<A, B>
{
    fn capture(&self, pipeline: &mut Pipeline) {
        let Self(a, b) = self;

        let dependencies = a
            .dependencies()
            .into_iter()
            .map(ControlFlow::Depend)
            .collect::<Vec<_>>();

        for Dependency { id, .. } in b.dependencies() {
            pipeline.control_flow(id).extend(dependencies.clone());
        }

        a.capture(pipeline);
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let Self(a, _) = self;
        a.dependencies()
    }
}

pub struct AfterMarker<Marker, Other>(PhantomData<(Marker, Other)>);

impl<Marker, Other, A: Sequence<Marker>, B: Sequence<Other>> Sequence<AfterMarker<Marker, Other>>
    for After<A, B>
{
    type Input = A::Input;
    type Output = A::Output;
}

impl<Marker, Other, A: Sequence<Marker>, B: Sequence<Other>> Capture<AfterMarker<Marker, Other>>
    for After<A, B>
{
    fn capture(&self, pipeline: &mut Pipeline) {
        let Self(a, b) = self;

        let dependencies = b
            .dependencies()
            .into_iter()
            .map(ControlFlow::Depend)
            .collect::<Vec<_>>();

        for Dependency { id, .. } in a.dependencies() {
            pipeline.control_flow(id).extend(dependencies.clone());
        }

        a.capture(pipeline);
    }

    fn dependencies(&self) -> Vec<Dependency> {
        let Self(a, _) = self;
        a.dependencies()
    }
}

pub struct SystemMarker<A, T>(PhantomData<(A, T)>);

pub struct BoxMarker<A, T>(PhantomData<(A, T)>);

impl<A: System, T: ToSystem<A>> Sequence<SystemMarker<A, T>> for T
where
    <A as System>::Input: Params + 'static,
    <A as System>::Output: Action + 'static,
{
    type Input = A::Input;
    type Output = A::Output;
}

impl<A: System, T: ToSystem<A>> Capture<SystemMarker<A, T>> for T
where
    <A as System>::Input: Params + 'static,
    <A as System>::Output: Action + 'static,
{
    fn capture(&self, pipeline: &mut Pipeline) {
        let job = SystemJob::new(self.to_system().boxed());
        pipeline.capture(Step {
            job: BoxedJob(Box::new(job), State::default()),
        })
    }
    fn dependencies(&self) -> Vec<Dependency> {
        vec![Dependency {
            id: self.id(),
            name: self.name().to_string(),
        }]
    }
}

#[derive(Debug, Default)]
pub struct State {
    sub_world: World,
    resources: Resources,
}

impl WorldState for State {
    fn spawn(&mut self) -> Entity {
        self.sub_world.spawn()
    }

    fn query<T: Fetch, F: Filter>(&mut self) -> Query<T, F> {
        self.sub_world.query()
    }

    fn insert(&mut self, entity: Entity, components: impl IntoComponents) -> Entity {
        self.sub_world.insert(entity, components)
    }
}

pub struct Return {
    state: State,
    action: Box<dyn Action>,
}

impl Return {
    pub(crate) fn act(
        mut self,
        dispatcher: &mut Dispatcher,
        world: &mut World,
        resources: &mut Resources,
    ) {
        self.action.act(
            dispatcher,
            &mut self.state.sub_world,
            &mut self.state.resources,
        );
        world.consume_state(self.state.sub_world);
        resources.consume_state(self.state.resources);
    }
    fn new(state: State, action: Box<dyn Action>) -> Self {
        Self { state, action }
    }
}

pub trait Job: System<Input = State, Output = Return> + fmt::Debug {
    fn state(&self, world: &mut World, resources: &mut Resources) -> State;
    fn boxed_job(&self) -> BoxedJob;
}

#[derive(Debug)]
pub struct BoxedJob(Box<dyn Job>, State);

impl Deref for BoxedJob {
    type Target = Box<dyn Job>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BoxedJob {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct SystemJob<Input, Output>(BoxedSystem<Input, Output>);

impl<'a, Input, Output> SystemJob<Input, Output> {
    pub fn new(mut sys: BoxedSystem<Input, Output>) -> Self {
        Self(sys)
    }
}

impl<'a, Input: 'a, Output: 'static> fmt::Debug for SystemJob<Input, Output> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(sys) = self;
        write!(f, "{}", sys.name())
    }
}

impl<Input: Params + 'static, Output: Action + 'static> System for SystemJob<Input, Output> {
    type Input = State;
    type Output = Return;

    fn boxed(&self) -> BoxedSystem<Self::Input, Self::Output> {
        let Self(sys) = self;
        BoxedSystem(Box::new(Self(sys.boxed())))
    }

    fn run(&mut self, mut state: Self::Input) -> Self::Output {
        let Self(sys) = self;
        dbg!(sys.name());
        let out = Box::new(sys.run(Params::new(&mut state)));
        Return::new(state, out)
    }

    fn id(&self) -> SystemId {
        let Self(sys) = self;
        sys.id()
    }

    fn name(&self) -> &str {
        let Self(sys) = self;
        sys.name()
    }
}

impl<Input: Params + 'static, Output: Action + 'static> Job for SystemJob<Input, Output> {
    fn state(&self, world: &mut World, resources: &mut Resources) -> State {
        let sub_world = world.sub_state(&Input::requires().into());
        let resources = resources.sub_state(&Input::requires().into());
        State {
            sub_world,
            resources,
        }
    }

    fn boxed_job(&self) -> BoxedJob {
        let Self(sys) = self;
        BoxedJob(Box::new(Self(sys.boxed())), State::default())
    }
}

pub trait Param {
    fn new(state: &mut State) -> Option<Self>
    where
        Self: Sized;
    fn requires() -> Prototype;
}

impl<'a, T: Fetch, F: Filter> Param for Query<T, F> {
    fn new(state: &mut State) -> Option<Self>
    where
        Self: Sized,
    {
        Some(state.query())
    }
    fn requires() -> Prototype {
        T::archetype().into()
    }
}

pub struct Res<T: Resource>(*const T);

impl<T: Resource> Deref for Res<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        interpret(T::identifier(), 0, self.0 as *const _)
    }
}

impl<T: Resource> Param for Res<T> {
    fn new(state: &mut State) -> Option<Self>
    where
        Self: Sized,
    {
        Some(Self(state.resources.get()?))
    }

    fn requires() -> Prototype {
        Prototype(
            [TypeAccess {
                data_id: DataId::Resource(T::identifier()),
                access: Access::Ref,
                size: mem::size_of::<T>(),
                name: type_name::<T>(),
            }]
            .into_iter()
            .collect(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResMut<T: Resource>(*mut T);

impl<T: Resource> Deref for ResMut<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        interpret(T::identifier(), 0, self.0 as *const _)
    }
}

impl<T: Resource> DerefMut for ResMut<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        interpret_mut(T::identifier(), 0, self.0 as *mut _)
    }
}

impl<T: Resource> Param for ResMut<T> {
    fn new(state: &mut State) -> Option<Self>
    where
        Self: Sized,
    {
        Some(Self(state.resources.get_mut()?))
    }

    fn requires() -> Prototype {
        Prototype(
            [TypeAccess {
                data_id: DataId::Resource(T::identifier()),
                access: Access::Mut,
                size: mem::size_of::<T>(),
                name: type_name::<T>(),
            }]
            .into_iter()
            .collect(),
        )
    }
}

impl<T: Param> Param for Option<T> {
    fn new(state: &mut State) -> Option<Self>
    where
        Self: Sized,
    {
        Some(T::new(state))
    }

    fn requires() -> Prototype {
        T::requires()
    }
}

pub trait Action {
    fn act(&mut self, dispatcher: &mut Dispatcher, world: &mut World, resources: &mut Resources);
}

impl_tuple_action!();

pub struct Insert<T: Resource>(pub Option<T>);

#[derive(Debug, Clone, Copy)]
pub enum Condition {
    Act,
    Defer,
}

impl From<bool> for Condition {
    fn from(value: bool) -> Self {
        value.then_some(Act).unwrap_or(Defer)
    }
}

impl Action for Condition {
    fn act(&mut self, dispatcher: &mut Dispatcher, world: &mut World, resources: &mut Resources) {
        dispatcher.set_active_condition(*self);
    }
}

impl FromResidual for Condition {
    fn from_residual(residual: <Self as Try>::Residual) -> Self {
        residual
    }
}

impl Try for Condition {
    type Output = Condition;
    type Residual = Condition;

    fn from_output(output: Self::Output) -> Self {
        output
    }

    fn branch(self) -> ops::ControlFlow<Self::Residual, Self::Output> {
        match self {
            Defer => ops::ControlFlow::Continue(self),
            Act => ops::ControlFlow::Break(self),
        }
    }
}

impl<T: Resource> From<T> for Insert<T> {
    fn from(value: T) -> Self {
        Self(Some(value))
    }
}

impl<T: Resource> Action for Insert<T> {
    fn act(&mut self, dispatcher: &mut Dispatcher, world: &mut World, resources: &mut Resources) {
        resources.insert(self.0.take().unwrap());
    }
}

pub trait Params {
    fn new(state: &mut State) -> Self
    where
        Self: Sized;
    fn requires() -> Prototype;
}

impl_tuple_param!();

pub struct Step {
    job: BoxedJob,
}

#[derive(Clone, Debug)]
pub struct Dependency {
    id: SystemId,
    name: String,
}

#[derive(Clone, Debug)]
pub enum ControlFlow {
    Depend(Dependency),
    If(Dependency),
    Staged,
}

#[derive(Default)]
pub struct Pipeline {
    steps: Vec<Step>,
    control_flow: BTreeMap<SystemId, Vec<ControlFlow>>,
}

impl Pipeline {
    pub(super) fn capture(&mut self, step: Step) {
        self.steps.push(step);
    }
    pub(super) fn control_flow(&mut self, id: SystemId) -> &mut Vec<ControlFlow> {
        self.control_flow.entry(id).or_default()
    }
}

pub trait ToPipeline<Marker> {
    fn to_pipeline(self) -> Pipeline;
}

impl<S: Sequence<Marker>, Marker> ToPipeline<Marker> for S {
    fn to_pipeline(self) -> Pipeline {
        let mut pipeline = Pipeline::default();
        self.capture(&mut pipeline);
        pipeline
    }
}

#[derive(Clone, Default, Debug)]
pub struct Logistics(Vec<ControlFlow>);

impl IntoIterator for Logistics {
    type Item = ControlFlow;
    type IntoIter = vec::IntoIter<ControlFlow>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Extend<ControlFlow> for Logistics {
    fn extend<T: IntoIterator<Item = ControlFlow>>(&mut self, iter: T) {
        self.0.extend(iter);
    }
}

impl Deref for Logistics {
    type Target = Vec<ControlFlow>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Logistics {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Default)]
pub struct Dispatcher {
    active: Option<SystemId>,
    conditions: BTreeMap<SystemId, Option<Condition>>,
    run_if: BTreeMap<SystemId, Vec<SystemId>>,
}

impl Dispatcher {
    pub(crate) fn set_active(&mut self, active: SystemId) {
        self.active = Some(active);
    }
    pub(crate) fn set_active_condition(&mut self, condition: Condition) {
        *self.conditions.get_mut(&self.active.unwrap()).unwrap() = Some(condition);
    }
}

#[derive(Default, Debug)]
pub struct Controller {
    topology: Vec<Task>,
}

#[derive(Default)]
pub struct Schedule {
    jobs: BTreeMap<SystemId, BoxedJob>,
    logic: BTreeMap<SystemId, Logistics>,
}

impl Schedule {
    pub fn run(&mut self, world: &mut World, resources: &mut Resources) {
        let (controller, mut dispatcher) = self.topology();
        'task: for mut task in controller.topology {
            if let Some(condition_systems) = dispatcher.run_if.get(&task.job.id()) {
                for id in condition_systems {
                    if let Some(Defer) = dispatcher.conditions.get(&id).map(|x| *x).flatten() {
                        continue 'task;
                    }
                }
            }
            let stage_value = resources.get::<ActiveStageValue>().map(|x| &x.0).unwrap();
            dispatcher.set_active(task.job.id());
            let state = task.job.state(world, resources);
            let ret = task.job.run(state);
            ret.act(&mut dispatcher, world, resources);
        }
    }

    fn topology(&mut self) -> (Controller, Dispatcher) {
        let mut controller = Controller::default();
        let mut dispatcher = Dispatcher::default();
        let systems = self.jobs.iter().map(|job| *job.0).collect::<BTreeSet<_>>();

        let mut logic = self.logic.clone();
        let mut logic_pending = BTreeMap::<SystemId, Logistics>::new();

        let mut logic_complete = systems.clone();
        systems
            .into_iter()
            .filter(|key| logic.contains_key(&key))
            .for_each(|key| {
                logic_complete.remove(&key);
            });

        while let Some(system_id) = logic_complete.pop_first() {
            controller.topology.push(Task {
                control_flow: logic_pending
                    .remove(&system_id)
                    .into_iter()
                    .flat_map(identity)
                    .collect::<Vec<_>>(),
                job: self.jobs[&system_id].boxed_job(),
            });

            for id in logic.keys().copied().collect::<Vec<_>>() {
                let logistics = logic.get_mut(&id).unwrap();
                let mut removed = vec![];

                for (index, control_flow) in logistics.iter().enumerate().rev() {
                    match control_flow {
                        ControlFlow::Depend(Dependency {
                            id: dependency_id, ..
                        }) => {
                            if *dependency_id == system_id {
                                removed.push((index, control_flow.clone()));
                            }
                        }
                        ControlFlow::If(Dependency {
                            id: dependency_id, ..
                        }) => {
                            if *dependency_id == system_id {
                                dispatcher.conditions.insert(system_id, None);
                                dispatcher.run_if.entry(id).or_default().push(system_id);
                                removed.push((index, control_flow.clone()));
                            }
                        }
                        ControlFlow::Staged => {}
                    }
                }

                for (index, control_flow) in removed {
                    logistics.remove(index);
                    logic_pending.entry(id).or_default().push(control_flow);
                }
                if logistics.is_empty() {
                    logic_complete.insert(id);
                    logic.remove(&id);
                }
            }
        }
        (controller, dispatcher)
    }

    pub fn add(&mut self, pipeline: Pipeline) {
        for (id, job) in pipeline
            .steps
            .into_iter()
            .map(|step| (step.job.id(), step.job))
        {
            self.jobs.insert(id, job);
        }

        for (id, cf) in pipeline.control_flow {
            self.logic.entry(id).or_default().extend(cf);
        }
    }
}

pub struct Task {
    control_flow: Vec<ControlFlow>,
    job: BoxedJob,
}

impl fmt::Debug for Task {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("Task");
        s.field("name", &self.job.name());
        s.field("control_flow", &self.control_flow);
        s.finish()
    }
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Resource, Clone, Copy)]
pub enum Section {
    Start,
    Update,
    Cleanup,
}

pub struct Staged<T: Stage>(T);

pub fn stage<T: Stage>(stage: T) -> Staged<T> {
    Staged(stage)
}

impl<T: Stage> Func<<Self as System>::Input, <Self as System>::Output> for Staged<T> {
    fn boxed(&self) -> Box<dyn Func<<Self as System>::Input, <Self as System>::Output>> {
        Box::new(Self(self.0))
    }

    fn execute(&mut self, input: <Self as System>::Input) -> <Self as System>::Output {
        let (value, section) = input;
        let stage = self.0;
        if let Some(value) = &(*value).0 {
            ((value.id == StageId(typeid::of::<T>())) && stage.active(*section)).into()
        } else {
            Defer
        }
    }
}

impl<T: Stage> System for Staged<T> {
    type Input = (ResMut<ActiveStageValue>, ResMut<Section>);
    type Output = Condition;

    fn boxed(&self) -> BoxedSystem<Self::Input, Self::Output> {
        BoxedSystem(Box::new(Self(self.0)))
    }

    fn run(&mut self, input: Self::Input) -> Self::Output {
        self.execute(input)
    }

    fn id(&self) -> SystemId {
        SystemId(typeid::of::<Self>())
    }

    fn name(&self) -> &str {
        type_name::<Self>()
    }
}

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub struct StageId(TypeId);

#[derive(Debug)]
pub struct StageValue {
    id: StageId,
}

#[derive(Resource, Debug)]
pub struct ActiveStageValue(Option<StageValue>);

pub trait Stage: Copy + 'static {
    fn active(&self, section: Section) -> bool;
    fn identifier() -> StageId {
        StageId(TypeId::of::<Self>())
    }
}

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub struct Start;
impl Stage for Start {
    fn active(&self, section: Section) -> bool {
        matches!(section, Section::Start)
    }
}
#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub struct Update;
impl Stage for Update {
    fn active(&self, section: Section) -> bool {
        matches!(section, Section::Update)
    }
}

pub type StageMax = usize;

#[derive(Default)]
pub struct Stages {
    stages: BTreeSet<StageId>,
}
impl Stages {
    fn register<T: Stage>(&mut self) {
        self.stages.insert(T::identifier());
    }
}

pub struct App {
    schedule: Schedule,
    world: World,
    resource: Resources,
    stages: Stages,
}

impl Default for App {
    fn default() -> Self {
        (Self {
            schedule: default(),
            world: World::with_partitions(128),
            resource: default(),
            stages: default(),
        })
        .stage::<Start>()
        .stage::<Update>()
    }
}

impl App {
    pub fn schedule<A>(mut self, sequence: impl Sequence<A>) -> Self {
        self.schedule.add(sequence.to_pipeline());
        self
    }
    pub fn insert<T: Resource>(mut self, resource: T) -> Self {
        self.resource.insert(resource);
        self
    }
    pub fn stage<T: Stage>(mut self) -> Self {
        self.stages.register::<T>();
        self
    }
    pub fn add(self, plugin: impl Plugin) -> Self {
        plugin.add(self)
    }
    pub fn run(mut self) -> ! {
        let schedule = Box::leak(Box::new(self.schedule));
        self.resource.insert(Section::Start);
        loop {
            self.resource.insert(ActiveStageValue(None));
            schedule.run(&mut self.world, &mut self.resource);

            for stage in &self.stages.stages {
                self.resource
                    .insert(ActiveStageValue(Some(StageValue { id: *stage })));
                schedule.run(&mut self.world, &mut self.resource);
            }
            self.resource.insert(Section::Update);
        }
    }
}

pub trait Plugin {
    fn add(self, app: App) -> App;
}
