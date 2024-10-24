use core::cmp::Ordering;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};

use crate::prelude::*;

#[derive(Default)]
pub struct World {
	entity_cursor: usize,
	entity_freed: Vec<Entity>,

	columns: BTreeMap<Archetype, ColumnState>,
}

#[derive(Clone, Copy)]
pub enum TakenMeta {
	InlineQuery {
		src_file: &'static str,
		src_line: usize,
	}
}

pub enum ColumnState {
	Avail(Column),
	Taken(TakenMeta)
}

impl Deref for ColumnState {
	type Target = Column;

	fn deref(&self) -> &Self::Target {
		self.unwrap_ref()
	}
}

impl DerefMut for ColumnState {
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.unwrap_mut()
	}
}

impl From<Column> for ColumnState {
	fn from(value: Column) -> Self {
		ColumnState::Avail(value)
	}
}

impl ColumnState {
	fn unwrap_mut(&mut self) -> &mut Column {
		if let Self::Avail(c) = self {
			return c;
		}
		panic!("column not available");
	}
	fn unwrap_ref(&self) -> &Column {
		if let Self::Avail(c) = self {
			return c;
		}
		panic!("column not available");
	}
	fn unwrap(self) -> Column {
		if let Self::Avail(c) = self {
			return c;
		}
		panic!("column not available");
	}
}

impl World {
	pub fn spawn(&mut self) -> Entity {
		let entity = Entity { identifier: self.entity_cursor, generation: 0 };
		self.entity_cursor += 1;
		entity
	}
	pub fn push(&mut self, components: impl IntoComponents) -> Entity {
		let entity = self.spawn();
		self.insert(entity, components)
	}
	pub fn extend<'a>(&'a mut self, components: impl IntoIterator<Item = impl IntoComponents + 'a> + 'a) -> impl Iterator<Item = Entity> + 'a {
		components.into_iter().map(|x| self.push(x))
	}
	pub fn insert(&mut self, entity: Entity, components: impl crate::ecs::IntoComponents) -> crate::ecs::Entity {
		let mut components = components.into_components();
		components.sort();
		let archetype = components.iter().map(|(meta, _)| meta).copied().collect::<Archetype>();
		let data = components.into_iter().flat_map(|(_, data)| data).collect::<Vec<_>>();
		let column = self.columns.entry(archetype.clone()).or_insert_with(|| ColumnState::from(Column::with_archetype(archetype.clone())));

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

#[derive(Ord, PartialOrd, Clone, Copy, Eq)]
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
			name: type_name::<Self>(),
			size: mem::size_of::<Self>()
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
		self.meta.iter().enumerate().find_map(|(i,x)| (x.identifier == p0).then_some(i)).unwrap()
	}
	pub(crate) fn is_superset(&self, other: &Archetype) -> bool {
		let id_super = self.meta.iter().map(|x| x.identifier).collect::<BTreeSet<_>>();
		let id_sub = other.meta.iter().map(|x| x.identifier).collect::<BTreeSet<_>>();

		id_sub.difference(&id_super).count() == 0
	}
}

impl PartialOrd for Archetype {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		self.meta.partial_cmp(&other.meta)
	}
}

impl FromIterator<ComponentMeta> for Archetype {
	fn from_iter<T: IntoIterator<Item=ComponentMeta>>(iter: T) -> Self {
		let meta = iter.into_iter().collect::<BTreeSet<_>>();
		let stride = (0..meta.len()).map(|i| meta.iter().take(i).map(|x| x.size).sum()).collect::<Vec<_>>();
		let size = meta.iter().map(|x| x.size).sum();
		Self {
			stride,
			meta,
			size}
	}
}

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub struct ComponentId(TypeId);

#[derive(Clone, Copy, Debug, Ord, Eq, PartialEq)]
pub struct ComponentMeta {
	identifier: ComponentId,
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
	archetype: Archetype
}

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
			archetype
		}
	}

	fn insert(&mut self, row: Row) -> Option<Row> {
		if let old = self.remove(row.entity) && old.is_some() {
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
		let data = self.data.splice(start..start + self.stride, iter::empty()).collect();
		Some(Row {
			entity,
			data,
			archetype: self.archetype.clone()
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
		Archetype::from_iter([T::meta()])
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

}

pub struct Passthrough;

impl Filter for Passthrough {}

pub struct Query<T: Fetch, F: Filter = Passthrough> {
	column: Vec<Column>,
	archetype: BTreeSet<Archetype>,
	components: PhantomData<T>,
	filter: F
}

impl<T: Fetch> Query<T, Passthrough> {
	pub fn new(mut world: World, meta: TakenMeta) -> Self {
		let mut take = vec![];

		let archetype_query = T::archetype();

		for archetype_column in world.columns.keys() {
			if archetype_column.is_superset(&archetype_query) {
				take.push(archetype_column.clone());
			}
		}

		let archetype = take.into_iter().collect::<BTreeSet<_>>();
		let column = archetype.iter().map(|archetype| world.columns.insert(archetype.clone(), ColumnState::Taken(meta)).expect("column does not exist by archetype")).map(ColumnState::unwrap).collect::<Vec<_>>();

		Self {
			column,
			archetype,
			components: PhantomData,
			filter: Passthrough
		}
	}
}

pub struct QueryIter<T: Fetch, F: Filter> {
	query: Query<T, F>,
	cursor_column: usize,
	cursor_row: usize,
	local_stride: Vec<Vec<usize>>,
}

impl<T: Fetch, F: Filter> IntoIterator for Query<T, F> {
	type Item = T::Item;
	type IntoIter = QueryIter<T, F>;

	fn into_iter(self) -> Self::IntoIter {
		let local_stride = self.archetype.iter().map(|a| T::local_stride(a)).collect();
		QueryIter {
			query: self,
			cursor_row: 0,
			cursor_column: 0,
			local_stride
		}
	}
}

impl<T: Fetch, F: Filter> Iterator for QueryIter<T, F> {
	type Item = T::Item;

	fn next(&mut self) -> Option<Self::Item> {
			if self.cursor_column >= self.query.column.len() {
				None?
			}

			let column = &mut self.query.column[self.cursor_column];

			if self.cursor_row >= column.len() {
				self.cursor_row = 0;
				self.cursor_column += 1;
			}

			let root = column.as_ptr(self.cursor_row);
			let item = Some(T::get(&mut 0, &self.local_stride[self.cursor_column], root));

			self.cursor_row += 1;

		item
	}
}

pub fn erase<T: 'static>(id: ComponentId, t: T) -> Vec<u8> {
	assert_eq!(id.0, TypeId::of::<T>());
	let t = ManuallyDrop::new(t);
	unsafe { ptr::slice_from_raw_parts(&t as *const _ as *const _, mem::size_of::<T>()).as_ref().unwrap().to_vec() }
}

pub fn interpret_mut<'a, T: 'static>(id: ComponentId, offset: usize, v: *mut u8) -> &'a mut T {
	assert_eq!(id.0, TypeId::of::<T>());
	unsafe { (v.add(offset) as *mut T).as_mut().unwrap() }
}