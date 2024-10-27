use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Query {
    Linear(usize),
    Position(Vector<3, usize>),
}

impl Query {
    pub(crate) fn index(self, structure: &dyn Volume) -> usize {
        match self {
            Query::Linear(i) => i,
            Query::Position(p) => structure.linearize(p),
        }
    }
}

pub trait Volume {
    fn size_axis(&self) -> Vector<3, usize>;
    fn count(&self) -> usize;

    fn get(&self, query: Query) -> Block;
    fn set(&mut self, query: Query, block: Block);

    fn linearize(&self, pos: Vector<3, usize>) -> usize {
        let Vector([x, y, z]) = pos;
        let Vector([x_max, y_max, _]) = self.size_axis();
        (z * x_max * y_max) + (y * x_max) + x
    }
    fn delinearize(&self, mut index: usize) -> Vector<3, usize> {
        let Vector([x_max, y_max, _]) = self.size_axis();
        let z = index / (x_max * y_max);
        index -= z * x_max * y_max;
        let y = index / x_max;
        let x = index % x_max;
        Vector([x, y, z])
    }
}

pub type BlockId = u64;

#[derive(Clone, Copy)]
pub struct Block(BlockId);

pub struct Structure {
    blocks: Vec<Block>,
    size_axis: Vector<3, usize>,
    transform: Transform,
}

impl Volume for Structure {
    fn size_axis(&self) -> Vector<3, usize> {
        self.size_axis
    }

    fn count(&self) -> usize {
        self.size_axis.resultant()
    }

    fn get(&self, query: Query) -> Block {
        let index = query.index(self);
        self.blocks[index]
    }

    fn set(&mut self, query: Query, block: Block) {
        let index = query.index(self);
        self.blocks[index] = block;
    }
}
