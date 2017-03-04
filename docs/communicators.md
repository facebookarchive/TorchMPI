# Communicators

In MPI, a communicator is the "communication universe" within which collective
and point-to-point communications operate.

A communicator in TorchMPI comprises 2 MPI::Intracomm objects which specify the
processes within(i.e. inner communicator) a communication context and
across(i.e. outer communicator) communication contexts.

## High-level API:

### torchmpi.communicatorNames()
Returns a string representation for the communicators the process belongs in.

## Low-level API:

### torchmpi.C.torchmpi_push_communicator(string)
```torchmpi.push_communicator(string)``` is a collective operation involving all
processes in the context of the current communicator. The current inner communicator
is decomposed into another TorchMPI communicator with outer/inner MPI::Intracomm
pairs. Processes that provide the same string to this function end up in the same
inner MPI::Intracomm.

If all inner MPI::Intracomm end up having the same size, the TorchMPI communicator
is said to be **cartesian**. Allreduce on a cartesian TorchMPI communicator can be
implemented in 2 steps: first Allreduce along the X-axis then the Y-axis.

Otherwise the TorchMPI communicator is said to have a **tree** structure. Allreduce
on a tree TorchMPI communicator is implemented in 3 steps: first Reduce to 0
within each inner communicator, then Allreduce across roots in the outer
communicator, then Broadcast from 0 within each inner communicator.

By default, MPI provides a global communicator that we explicitly duplicate with
the 'global' string, it contains all the processes.
TorchMPI further subdivides processes by hostnames and by cudaIPC communication
capability (if relevant).

### torchmpi.C.torchmpi_set_communicator(int level)
Set the current context to the TorchMPI communicator at the specified level. This
is a collective call that needs to be called by all processes in the current
communicator.

### torchmpi.C.torchmpi_is_cartesian_communicator()
Returns whether the communicator is cartesian of not (i.e. tree).

### torchmpi.C.torchmpi_num_nodes_in_communicator()
Returns the number of nodes in the current inner communicator.

### torchmpi.C.torchmpi_set_tree_communicator()
Switch to building only tree communicators which will trigger 3-steps Allreduce.
This is a collective call that needs to be called by all processes in the current
communicator.
Must be called before the target communicator is created
(i.e. before torchmpi.start() in the case of the default communicators).

### torchmpi.C.torchmpi_set_cartesian_communicator()
Switch to building cartesian communicators (default mode). This is best effort;
if the communicator does not have a cartesian structure, it will remain a tree.
This is a collective call that needs to be called by all processes in the current
communicator.
Must be called before the target communicator is created
(i.e. before torchmpi.start() in the case of the default communicators).