"""data/: Data partitioning strategies and dataset loaders."""
from flsim.data.iid import IIDPartitioner
from flsim.data.shard import ShardPartitioner
from flsim.data.dirichlet import DirichletPartitioner

__all__ = ["IIDPartitioner", "ShardPartitioner", "DirichletPartitioner"]
