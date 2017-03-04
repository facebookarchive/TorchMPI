package = "torchmpi"
version = "scm-1"

source = {
   url = "git://github.com/nicolasvasilache/torchmpi.git"
}

description = {
   summary = "MPI for Torch",
   detailed = [[
      Various abstractions for baseline distributed torch.
   ]],
   homepage = "https://github.com/nicolasvasilache/torchmpi",
   license = "BSD"
}

dependencies = {
   "torch",
   -- dependencies for the apps:
   "torchnet",
   "mnist",
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)",
      MPI_CXX_COMPILER="${MPI_CXX_COMPILER}",
      CMAKE_PREFIX_PATH="$(LUA_BINDIR)/..", -- to find torch
   }
}
