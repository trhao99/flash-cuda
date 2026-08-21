<!-- source: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html -->
# 2. Profiling Guide

Nsight Compute profiling guide.

Profiling Guide with metric types and meaning, data collection modes and FAQ for common problems.

## 2.1. Introduction

This guide describes various profiling topics related to NVIDIA Nsight Compute and NVIDIA Nsight Compute CLI. Most of these apply to both the UI and the CLI version of the tool.

To use the tools effectively, it is recommended to read this guide, as well as at least the following chapters of the *CUDA Programming Guide*:

- [Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)

- [Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)

- [Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidelines)

Afterwards, it should be enough to read the *Quickstart* chapter of the NVIDIA Nsight Compute or NVIDIA Nsight Compute CLI documentation, respectively, to start using the tools.

### 2.1.1. Profiling Applications

During regular execution, a CUDA application process will be launched by the user. It communicates directly with the CUDA user-mode driver, and potentially with the CUDA runtime library.



Regular Application Execution

When profiling an application with NVIDIA Nsight Compute, the behavior is different. The user launches the NVIDIA Nsight Compute frontend (either the UI or the CLI) on the host system, which in turn starts the actual application as a new process on the target system. While host and target are often the same machine, the target can also be a remote system with a potentially different operating system.

The tool inserts its measurement libraries into the application process, which allow the profiler to intercept communication with the CUDA user-mode driver. In addition, when a kernel launch is detected, the libraries can collect the requested performance metrics from the GPU. The results are then transferred back to the frontend.



Profiled Application Execution

## 2.2. Metric Collection

Collection of performance metrics is the key feature of NVIDIA Nsight Compute. Since there is a huge list of metrics available, it is often easier to use some of the tool’s pre-defined [sets or sections](index.html#sets-and-sections) to collect a commonly used subset. Users are free to adjust which metrics are collected for which kernels as needed, but it is important to keep in mind the [Overhead](index.html#overhead) associated with data collection.

### 2.2.1. Sets and Sections

NVIDIA Nsight Compute uses *Section Sets* (short *sets*) to decide, on a very high level, the number of metrics to be collected. Each set includes one or more *Sections*, with each section specifying several logically associated metrics. For example, one section might include only high-level SM and memory utilization metrics, while another could include metrics associated with the memory units, or the HW scheduler.

The number and type of metrics specified by a section has significant impact on the overhead during profiling. To allow you to quickly choose between a fast, less detailed profile and a slower, more comprehensive analysis, you can select the respective section set. See [Overhead](index.html#overhead) for more information on profiling overhead.

By default, a relatively small number of metrics is collected. Those mostly include high-level utilization information as well as static launch and occupancy data. The latter two are regularly available without replaying the kernel launch. The `basic` set is collected when no `--set`, `--section` and no `--metrics` options are passed on the command line. The full set of sections can be collected with `--set`` ``full`.

Use `--list-sets` to see the list of currently available sets. Use `--list-sections` to see the list of currently available sections. The default search directory and the location of pre-defined section files are also called `sections/`. All related command line options can be found in the NVIDIA Nsight Compute CLI documentation.

A file named `.ncu-ignore` may be placed in any directory to have its contents ignored when the tool looks for section (and rule) files. When adding section directories [recursively](../NsightComputeCli/index.html#command-line-options-profile), even if the file is present, sub-directories are still searched.

### 2.2.2. Sections and Rules

| Identifier and Filename | Description |
|----|----|
| ComputeWorkloadAnalysis (Compute Workload Analysis) | Detailed analysis of the compute resources of the streaming multiprocessors (SM), including the achieved instructions per clock (IPC) and the utilization of each available pipeline. Pipelines with very high utilization might limit the overall performance. |
| InstructionStats (Instruction Statistics) | Statistics of the executed low-level assembly instructions (SASS). The instruction mix provides insight into the types and frequency of the executed instructions. A narrow mix of instruction types implies a dependency on few instruction pipelines, while others remain unused. Using multiple pipelines allows hiding latencies and enables parallel execution. |
| LaunchStats (Launch Statistics) | Summary of the configuration used to launch the kernel. The launch configuration defines the size of the kernel grid, the division of the grid into blocks, and the GPU resources needed to execute the kernel. Choosing an efficient launch configuration maximizes device utilization. |
| MemoryWorkloadAnalysis (Memory Workload Analysis) | Detailed analysis of the memory resources of the GPU. Memory can become a limiting factor for the overall kernel performance when fully utilizing the involved hardware units (Mem Busy), exhausting the available communication bandwidth between those units (Max Bandwidth), or by reaching the maximum throughput of issuing memory instructions (Mem Pipes Busy). Depending on the limiting factor, the memory chart and tables allow to identify the exact bottleneck in the memory system. |
| NUMA Affinity (NumaAffinity) | Non-uniform memory access (NUMA) affinities based on compute and memory distances for all GPUs. |
| Nvlink (Nvlink) | High-level summary of NVLink utilization. It shows the total received and transmitted (sent) memory, as well as the overall link peak utilization. |
| Nvlink_Tables (Nvlink_Tables) | Detailed tables with properties for each NVLink. |
| Nvlink_Topology (Nvlink_Topology) | NVLink Topology diagram shows logical NVLink connections with transmit/receive throughput. |
| Occupancy (Occupancy) | Occupancy is the ratio of the number of active warps per multiprocessor to the maximum number of possible active warps. Another way to view occupancy is the percentage of the hardware’s ability to process warps that is actively in use. Higher occupancy does not always result in higher performance, however, low occupancy always reduces the ability to hide latencies, resulting in overall performance degradation. Large discrepancies between the theoretical and the achieved occupancy during execution typically indicates highly imbalanced workloads. |
| PM Sampling (PmSampling) | Timeline view of metrics sampled periodically over the workload duration. Data is collected across multiple passes. Use this section to understand how workload behavior changes over its runtime. |
| PM Sampling: Warp States (PmSampling_WarpStates) | Warp states sampled periodically over the workload duration. Metrics in different groups come from different passes. |
| SchedulerStats (Scheduler Statistics) | Summary of the activity of the schedulers issuing instructions. Each scheduler maintains a pool of warps that it can issue instructions for. The upper bound of warps in the pool (Theoretical Warps) is limited by the launch configuration. On every cycle each scheduler checks the state of the allocated warps in the pool (Active Warps). Active warps that are not stalled (Eligible Warps) are ready to issue their next instruction. From the set of eligible warps, the scheduler selects a single warp from which to issue one or more instructions (Issued Warp). On cycles with no eligible warps, the issue slot is skipped and no instruction is issued. Having many skipped issue slots indicates poor latency hiding. |
| SourceCounters (Source Counters) | Source metrics, including branch efficiency and sampled warp stall reasons. Warp Stall Sampling metrics are periodically sampled over the kernel runtime. They indicate when warps were stalled and couldn’t be scheduled. See the documentation for a description of all stall reasons. Only focus on stalls if the schedulers fail to issue every cycle. |
| SpeedOfLight (GPU Speed Of Light Throughput) | High-level overview of the throughput for compute and memory resources of the GPU. For each unit, the throughput reports the achieved percentage of utilization with respect to the theoretical maximum. Breakdowns show the throughput for each individual sub-metric of Compute and Memory to clearly identify the highest contributor. |
| Tile (Tile Statistics) | Tile kernel launch configuration, execution information and GPU resource usage. Tile kernels are translated into Tile IR (Tile Intermediate Representation), an extension to the CUDA programming model that enables first-class support for tile programming. |
| WarpStateStats (Warp State Statistics) | Analysis of the states in which all warps spent cycles during the kernel execution. The warp states describe a warp’s readiness or inability to issue its next instruction. The warp cycles per instruction define the latency between two consecutive instructions. The higher the value, the more warp parallelism is required to hide this latency. For each warp state, the chart shows the average number of cycles spent in that state per issued instruction. Stalls are not always impacting the overall performance nor are they completely avoidable. Only focus on stall reasons if the schedulers fail to issue every cycle. |

Available Sections {#id28 .table-no-stripes .docutils .align-default}

### 2.2.3. Replay

Depending on which metrics are to be collected, kernels might need to be *replayed* one or more times, since not all metrics can be collected in a single *pass*. For example, the number of metrics originating from hardware (HW) performance counters that the GPU can collect at the same time is limited. In addition, patch-based software (SW) performance counters can have a high impact on kernel runtime and would skew results for HW counters.

#### Kernel Replay

In *Kernel Replay*, all metrics requested for a specific kernel instance in NVIDIA Nsight Compute are grouped into one or more passes. For the first pass, all GPU memory that can be accessed by the kernel is saved. After the first pass, the subset of memory that is written by the kernel is determined. Before each pass (except the first one), this subset is restored in its original location to have the kernel access the same memory contents in each replay pass.

NVIDIA Nsight Compute attempts to use the fastest available storage location for this save-and-restore strategy. For example, if data is allocated in device memory, and there is still enough device memory available, it is stored there directly. If it runs out of device memory, the data is transferred to the CPU host memory. Likewise, if an allocation originates from CPU host memory, the tool first attempts to save it into the same memory location, if possible.

As explained in [Overhead](index.html#overhead), the time needed for this increases the more memory is accessed, especially written, by a kernel. If NVIDIA Nsight Compute determines that only a single replay pass is necessary to collect the requested metrics, no save-and-restore is performed at all to reduce overhead.



Regular Application Execution



Execution with Kernel Replay. All memory is saved, and memory written by the kernel is restored in-between replay passes.

#### Application Replay

In *Application Replay*, all metrics requested for a specific kernel launch in NVIDIA Nsight Compute are grouped into one or more passes. In contrast to [Kernel Replay](index.html#kernel-replay), the complete application is run multiple times, so that in each run one of those passes can be collected per kernel.

For correctly identifying and combining performance counters collected from multiple application replay passes of a single kernel launch into one result, the application needs to be deterministic with respect to its kernel activities and their assignment to GPUs, contexts, streams, and potentially NVTX ranges. Normally, this also implies that the application needs to be deterministic with respect to its overall execution.

Application replay has the benefit that memory accessed by the kernel does not need to be saved and restored via the tool, as each kernel launch executes only once during the lifetime of the application process. Besides avoiding memory save-and-restore overhead, application replay also allows to disable [Cache Control](index.html#cache-control). This is especially useful if other GPU activities preceding a specific kernel launch are used by the application to set caches to some expected state.

In addition, application replay can support profiling kernels that have interdependencies to the host during execution. With kernel replay, this class of kernels typically hangs when being profiled, because the necessary responses from the host are missing in all but the first pass. In contrast, application replay ensures the correct behavior of the program execution in each pass.

In contrast to kernel replay, multiple passes collected via application replay imply that all host-side activities of the application are duplicated, too. If the application requires significant time for e.g. setup or file-system access, the overhead will increase accordingly.



Regular Application Execution



Execution with Application Replay. No memory is saved or restored, but the cost of running the application itself is duplicated.

Across application replay passes, NVIDIA Nsight Compute matches metric data for the individual, selected kernel launches. The matching strategy can be selected using the `--app-replay-match` option. For matching, only kernels within the same process and running on the same device are considered. By default, the *grid* strategy is used, which matches launches according to their kernel name and grid size. When multiple launches have the same attributes (e.g. name and grid size), they are matched in execution order.



Kernel matching during application replay using the *grid* strategy.

#### Range Replay

In *Range Replay*, all requested metrics in NVIDIA Nsight Compute are grouped into one or more passes. In contrast to [Kernel Replay](index.html#kernel-replay) and [Application Replay](index.html#application-replay), *Range Replay* captures and replays complete ranges of CUDA API calls and kernel launches within the profiled application. Metrics are then not associated with individual kernels but with the entire range. This allows the tool to execute kernels without serialization and thereby supports profiling kernels that should be run concurrently for correctness or performance reasons.



Execution with Range Replay. An entire range of API calls and kernel launches is captured and replayed. Host and device memory is saved and restored as necessary.

##### Defining Ranges

Range replay requires you to specify the range for profiling in the application. A range is defined by a start and an end marker and includes all CUDA API calls and kernels launched between these markers from any CPU thread. The application is responsible for inserting appropriate synchronization between threads to ensure that the anticipated set of API calls is captured. Range markers can be set using one of the following options:

- **Profiler Start/Stop API**

  Set the start marker using `cu(da)ProfilerStart` and the end marker using `cu(da)ProfilerStop`. Note: The CUDA driver API variants of this API require to include `cudaProfiler.h`. The CUDA runtime variants require to include `cuda_profiler_api.h`.

  This is the default for NVIDIA Nsight Compute.

- **NVTX Ranges**

  Define the range using an [NVTX Include](../NsightComputeCli/index.html#nvtx-filtering) expression. The range capture starts with the first CUDA API call and ends at the last API call for which the expression is matched, respectively. If multiple expressions are specified, a range is defined as soon as any of them matches. Hence, multiple expressions can be used to conveniently capture and profile multiple ranges for the same application execution.

  The application must have been instrumented with the NVTX API for any expressions to match.

  This mode is enabled by passing `--nvtx`` ``--nvtx-include`` ``<expression>`` ``[--nvtx-include`` ``<expression>]` to the NVIDIA Nsight Compute CLI.

Ranges must fulfill several requirements:

- It must be possible to synchronize all active CUDA contexts at the start of the range.

- Ranges must not include unsupported CUDA API calls. See [Supported APIs](index.html#range-replay-supported-apis) for the list of currently supported APIs.

In addition, there are several recommendations that ranges should comply with to guarantee a correct capture and replay:

- Set ranges as narrow as possible for capturing a specific set of CUDA launches. The more API calls are included, the higher the potentially created overhead from capturing and replaying these API calls.

- Avoid freeing host allocations written by device memory during the range. This includes both heap as well as stack allocations. NVIDIA Nsight Compute does not intercept creation or destruction of generic host (CPU)-based allocations. However, to guarantee correct program execution after any replay of the range, the tool attempts to restore host allocations that were written from device memory during the capture. If these host addresses are invalid or re-assigned, the program behavior is undefined and potentially unstable. In cases where avoiding freeing such allocations is not possible, you should limit profiling to one range using `--launch-count`` ``1`, set the *disable-host-restore* range replay option and optionally use `--kill`` ``yes` to terminate the process after this range.

- Avoid updating device-mapped host memory in between dependent kernel launches. NVIDIA Nsight Compute does not capture host memory state for every kernel which is launched in between dependent kernel launches.

- Defining narrow ranges at local scope helps with avoiding destruction of stack allocations during the range, too.

- When defining the range markers using `cu(da)ProfilerStart/Stop`, prefer the CUDA driver API calls `cuProfilerStart/Stop`. Internally, NVIDIA Nsight Compute only intercepts the CUDA driver API variants and the CUDA runtime API may not trigger these if no CUDA context is active on the calling thread.

##### Supported APIs

Range replay supports a subset of the CUDA API for capture and replay. This page lists the supported functions as well as any further, API-specific limitations that may apply. If an unsupported API call is detected in the captured range, an error is reported and the range cannot be profiled. The groups listed below match the ones found in the [CUDA Driver API documentation](https://docs.nvidia.com/cuda/cuda-driver-api/index.html).

Generally, range replay only captures and replay CUDA *Driver* API calls. CUDA *Runtime* APIs calls can be captured when they generate only supported CUDA Driver API calls internally. Deprecated APIs are not supported.

**Error Handling**

All supported.

**Initialization**

Not supported.

**Version Management**

All supported.

**Device Management**

All supported, except:

- cuDeviceSetMemPool

**Primary Context Management**

- cuDevicePrimaryCtxGetState

**Context Management**

All supported, except:

- cuCtxSetCacheConfig

- cuCtxSetSharedMemConfig

**Module Management**

- cuModuleGetFunction

- cuModuleGetGlobal

- cuModuleGetSurfRef

- cuModuleGetTexRef

- cuModuleLoad

- cuModuleLoadData

- cuModuleLoadDataEx

- cuModuleLoadFatBinary

- cuModuleUnload

**Library Management**

All supported, except:

- cuKernelSetAttribute

- cuKernelSetCacheConfig

**Memory Management**

- cuArray\*

- cuDeviceGetByPCIBusId

- cuDeviceGetPCIBusId

- cuMemAlloc

- cuMemAllocHost

- cuMemAllocPitch

- cuMemBatchDecompressAsync

- cuMemcpy\*

- cuMemFree

- cuMemFreeHost

- cuMemGetAddressRange

- cuMemGetInfo

- cuMemHostAlloc

- cuMemHostGetDevicePointer

- cuMemHostGetFlags

- cuMemHostRegister

- cuMemHostUnregister

- cuMemset\*

- cuMipmapped\*

**Virtual Memory Management**

Not supported.

**Stream Ordered Memory Allocator**

Not supported.

**Unified Addressing**

Not supported.

**Stream Management**

- cuStreamCreate\*

- cuStreamDestroy

- cuStreamGet\*

- cuStreamQuery

- cuStreamSetAttribute

- cuStreamSynchronize

- cuStreamWaitEvent

**Event Management**

All supported.

**External Resource interoperability**

Not supported.

**Stream Memory Operations**

Not supported.

**Execution Control**

- cuFuncGetAttribute

- cuFuncGetModule

- cuFuncSetAttribute

- cuFuncSetCacheConfig

- cuLaunchCooperativeKernel

- cuLaunchHostFunc

- cuLaunchKernel

**Graph Management**

Not supported.

**Occupancy**

All supported.

**Texture/Surface Reference Management**

Not supported.

**Texture Object Management**

All supported.

**Surface Object Management**

All supported.

**Peer Context Memory Access**

Not supported.

**Graphics Interoperability**

Not supported.

**Driver Entry Point Access**

All supported.

**Surface Object Management**

All supported.

**OpenGL Interoperability**

Not supported.

**VDPAU Interoperability**

Not supported.

**EGL Interoperability**

Not supported.

**Green Contexts**

- cuCtxFromGreenCtx

- cuGreenCtxCreate

- cuGreenCtxDestroy

- cuGreenCtxRecordEvent

- cuGreenCtxStreamCreate

- cuGreenCtxWaitEvent

- cuStreamGetDevResource

- cuStreamGetGreenCtx

#### Application Range Replay

In *Application Range Replay*, all requested metrics in NVIDIA Nsight Compute are grouped into one or more passes. Similar to [Range Replay](index.html#range-replay), metrics are not associated with individual kernels but with the entire selected range. This allows the tool to execute workloads (kernels, CUDA graphs, …) without serialization and thereby supports profiling workloads that must be run concurrently for correctness or performance reasons.

In contrast to Range Replay, the range is not explicitly captured and executed directly for each pass, but instead the entire application is re-run multiple times, with one pass collected for each range in every application execution. This has the benefit that no application state must be observed and captured for each range and API calls within the range do not need to be supported explicitly, as correct execution of the range is handled by the application itself.

Defining ranges to profile is identical to [Range Replay](index.html#range-replay-define-range). The CUDA context for which the range should be profiled must be current to the thread defining the start of the range and must be active for the entire range.



Execution with Application Range Replay. A range of workloads is replayed by re-running the entire application without modifying interactions or saving and restoring memory.

#### Limitations

- In *Application Range Replay*, instruction-level SASS metrics do not include profiling data for JIT-compiled kernels from the range.

#### Graph Profiling

In multiple replay modes, NVIDIA Nsight Compute can profile CUDA graphs as single workload entities, rather than profile individual kernel nodes. The behavior can be toggled in the respective [command line](../NsightComputeCli/index.html#command-line-options-profile) or [UI](../NsightCompute/index.html#connection-activity-interactive) options.

The primary use cases for enabling this mode are:

- Profile graphs that include mandatory concurrent kernel nodes.

- Profile graph behavior more accurately across multiple kernel node launches, as caches are not purged in between nodes.

Additional notes:

- Individual kernel nodes that launch device-sided graphs can be profiled excluding the launched graphs. Requires NVIDIA display driver version 580 or higher.

- Device-launched graphs are supported in node-level graph profiling mode. Note that any pending updates to the instantiated graph (not yet uploaded to the device) will be applied during profiling. Requires driver version 590 or higher.

- Conditional graph nodes are supported in node-level graph profiling mode. Requires driver version 590 or higher.

- When graph profiling is enabled, certain metrics such as instruction-level source metrics are not available. This then also applies to kernels profiled outside of graphs.

### 2.2.4. Compatibility

The set of available [replay modes](index.html#replay) and [metrics](index.html#metrics-guide) depends on the type of GPU workload to profile.

|  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|
| Workload Type | Replay Mode |  |  |  | Metric Groups |  |  |  |  |
|  | Kernel | Application | Range | Application-Range | Hardware Counters / SMSP | Unit-Level Source | Instruction-Level Source [3](#fmetricssass1) | Launch | Warp/PM Sampling |
| Kernel | Yes | Yes | Yes [2](#fcompat2) | Yes [2](#fcompat2) | Yes | Yes | Yes | Yes | Yes |
| Range | No | No | Yes | Yes | Yes | No | Yes | Some | Yes |
| Cmdlist | Yes | No | No | No | Yes | Yes | Yes | Some | Yes |
| Graph [1](#fcompat1) | Yes | No | No | No | Yes | No | No | Some | Yes |

Replay modes and metric compatibility per workload type {#id36 .table-no-stripes .docutils .align-default}

Footnotes

[1](#id4)  
Limitations also apply to kernels profiled outside of graphs.

2([1](#id2),[2](#id3))  
Workload type is supported as part of the profiled range, but not separated in the result. Metric support matches that of Range workloads.

[3](#id1)  
Instruction-level source metrics do not require profiling permissions on the target device when collected through the command line interface.

### 2.2.5. Profile Series

The performance of a kernel is highly dependent on the used launch parameters. Small changes to the launch parameters can have a significant effect on the runtime behavior of the kernel. However, identifying the best parameter set for a kernel by manually testing a lot of combinations can be a tedious process.

To make this workflow faster and more convenient, Profile Series provide the ability to automatically profile a single kernel multiple times with changing parameters. The parameters to be modified and values to be tested can be independently enabled and configured. For each combination of selected parameter values a unique profile result is collected. And the modified parameter values are tracked in the description of the results of a series. By comparing the results of a profile series, the kernel’s behavior on the changing parameters can be seen and the most optimal parameter set can be identified quickly.



Profile Series action.



Profile Series dialog.

### 2.2.6. Overhead

As with most measurements, collecting performance data using NVIDIA Nsight Compute CLI incurs some runtime overhead on the application. The overhead does depend on a number of different factors:

- **Number and type of collected metrics**

  Depending on the selected metric, data is collected either through a hardware performance monitor on the GPU, through software patching of the kernel instructions or via a launch or device attribute. The overhead between these mechanisms varies greatly, with launch and device attributes being “statically” available and requiring no kernel runtime overhead. The overhead incurred by hardware metrics is very low.

  Software-patched metrics incur the highest overhead, as they require the kernel to be modified and all patched instructions to execute additional code. The exact overhead depends on the number and type of software-patched metrics being collected. It is important though that software-patched metrics are collected in separate replay passes and therefore don’t interfere with the correctness or precision of other metrics. Their values are also independent of any overhead they may cause.

  Furthermore, only a limited number of metrics can be collected in a single *pass* of the kernel execution. If more metrics are requested, the kernel launch is *replayed* multiple times, with its accessible memory being saved and restored between subsequent passes to guarantee deterministic execution. Therefore, collecting more metrics can significantly increase overhead by requiring more replay passes and increasing the total amount of memory that needs to be restored during replay.

- **The collected section set**

  Since each [set](index.html#sets-and-sections) specifies a group of sections to be collected, choosing a less comprehensive set can reduce profiling overhead. See the `--set` command in the [NVIDIA Nsight Compute CLI](../NsightComputeCli/index.html#command-line-options-profile) documentation.

- **Number of collected sections**

  Since each [section](index.html#sets-and-sections) specifies a number of metrics to be collected, selecting fewer sections can reduce profiling overhead. See the `--section` command in the [NVIDIA Nsight Compute CLI](../NsightComputeCli/index.html#command-line-options-profile) documentation.

- **Number of profiled kernels**

  By default, all selected metrics are collected for all launched kernels. To reduce the impact on the application, you can try to limit performance data collection to as few kernel functions and instances as makes sense for your analysis. See the filtering commands in the [NVIDIA Nsight Compute CLI](../NsightComputeCli/index.html#command-line-options-profile) documentation.

  There is a relatively high one-time overhead for the first profiled kernel in each context to generate the metric configuration. This overhead does not occur for subsequent kernels in the same context, if the list of collected metrics remains unchanged.

- **GPU Architecture**

  For some metrics, the overhead can vary depending on the exact chip they are collected on, e.g. due to varying number of units on the chip. Similarly, the overhead for resetting the L2 cache in-between kernel replay passes depends on the size of that cache.

## 2.3. Metrics Guide

### 2.3.1. Hardware Model

#### Compute Model

All NVIDIA GPUs are designed to support a general purpose heterogeneous parallel programming model, commonly known as *Compute*. This model decouples the GPU from the traditional graphics pipeline and exposes it as a general purpose parallel multi-processor. A heterogeneous computing model implies the existence of a host and a device, which in this case are the CPU and GPU, respectively. At a high level view, the host (CPU) manages resources between itself and the device and will send work off to the device to be executed in parallel.

Central to the compute model is the Grid, Block, Thread hierarchy, which defines how compute work is organized on the GPU. The hierarchy from top to bottom is as follows:

- A *Grid* is a 1D, 2D or 3D array of thread blocks.

- A *Block* is a 1D, 2D or 3D array of threads, also known as a *Cooperative Thread Array (CTA)*.

- A *Thread* is a single thread which runs on one of the GPU’s SM units.

The purpose of the Grid, Block, Thread hierarchy is to expose a notion of locality amongst a group of threads, i.e. a Cooperative Thread Array (CTA). In CUDA, CTAs are referred to as Thread Blocks. The architecture can exploit this locality by providing fast shared memory and barriers between the threads within a single CTA. When a Grid is launched, the architecture guarantees that all threads within a CTA will run concurrently on the same SM. Information on the grids and blocks can be found in the [Launch Statistics](index.html#sections-and-rules) section.

The number of CTAs that fit on each SM depends on the physical resources required by the CTA. These resource limiters include the number of threads and registers, shared memory utilization, and hardware barriers. The number CTAs per SM is referred to as the CTA *occupancy*, and these physical resources limit this occupancy. Details on the kernel’s occupancy are collected by the [Occupancy](index.html#sections-and-rules) section.

Each CTA can be scheduled on any of the available SMs, where there is no guarantee in the order of execution. As such, CTAs must be entirely independent, which means it is not possible for one CTA to wait on the result of another CTA. As CTAs are independent, the host (CPU) can launch a large Grid that will not fit on the hardware all at once, however any GPU will still be able to run it and produce the correct results.

CTAs are further divided into groups of 32 threads called *Warps*. If the number of threads in a CTA is not dividable by 32, the last warp will contain the remaining number of threads.

The total number of CTAs that can run concurrently on a given GPU is referred to as *Wave*. Consequently, the size of a Wave scales with the number of available SMs of a GPU, but also with the occupancy of the kernel.

#### Streaming Multiprocessor

The *Streaming Multiprocessor (SM)* is the core processing unit in the GPU. The SM is optimized for a wide diversity of workloads, including general-purpose computations, deep learning, ray tracing, as well as lighting and shading. The SM is designed to simultaneously execute multiple CTAs. CTAs can be from different grid launches.

The SM implements an execution model called Single Instruction Multiple Threads (SIMT), which allows individual threads to have unique control flow while still executing as part of a warp. The Turing SM inherits the Volta SM’s independent thread scheduling model. The SM maintains execution state per thread, including a program counter (PC) and call stack. The independent thread scheduling allows the GPU to yield execution of any thread, either to make better use of execution resources or to allow a thread to wait for data produced by another thread possibly in the same warp. Collecting the [Source Counters](index.html#sections-and-rules) section allows you to inspect instruction execution and predication details on the *Source Page*, along with [Sampling](index.html#sampling) information.

Each SM is partitioned into four processing blocks, called *SM sub partitions*. The SM sub partitions are the primary processing elements on the SM. Each sub partition contains the following units:

- Warp Scheduler

- Register File

- Execution Units/Pipelines/Cores

  - Integer Execution units

  - Floating Point Execution units

  - Memory Load/Store units

  - Special Function unit

  - Tensor Cores

Shared within an SM across the four SM partitions are:

- Unified L1 Data Cache / Shared Memory

- Texture units

- RT Cores, if available

A warp is allocated to a sub partition and resides on the sub partition from launch to completion. A warp is referred to as *active* or *resident* when it is mapped to a sub partition. A sub partition manages a fixed size pool of warps. On Volta architectures, the size of the pool is 16 warps. On Turing architectures the size of the pool is 8 warps. Active warps can be in *eligible* state if the warp is ready to issue an instruction. This requires the warp to have a decoded instruction, all input dependencies resolved, and for the function unit to be available. Statistics on active, eligible and issuing warps can be collected with the [Scheduler Statistics](index.html#sections-and-rules) section.

A warp is *stalled* when the warp is waiting on

- an instruction fetch,

- a memory dependency (result of memory instruction),

- an execution dependency (result of previous instruction), or

- a synchronization barrier.

See [Warp Scheduler States](index.html#statistical-sampler) for the list of stall reasons that can be profiled and the [Warp State Statistics](index.html#sections-and-rules) section for a summary of warp states found in the kernel execution.

The most important resource under the compiler’s control is the number of registers used by a kernel. Each sub partition has a set of 32-bit registers, which are allocated by the HW in fixed-size chunks. The [Launch Statistics](index.html#sections-and-rules) section shows the kernel’s register usage.

#### Compute Preemption

[Compute preemption](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#compute-preemption) avoids long running kernels from monopolizing the GPU, at the risk of context switch overhead. The execution context (registers, shared memory, etc.) is saved at preemption and restored later. Context switches happen at instruction-level granularity. The [exclusive-process](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-modes) compute mode may be used on supported systems to avoid context switching.

#### Memory

NVIDIA GPUs typically utilize several different types of memory, including global, local, shared, constant, texture, and surface memory, as well as registers. Each of these memory types has different characteristics, including size, latency, and access patterns. The following paragraphs describe some of these different types of memory.

*Global memory* is a 49-bit virtual address space that is mapped to physical memory on the device, pinned system memory, or peer memory. Global memory is visible to all threads in the GPU. Global memory is accessed through the SM L1 and GPU L2 caches (see below).

*Local memory* is thread-private memory: each CUDA thread has its own dedicated portion of local memory. Local memory accesses only occur for automatic variables (i.e., variables declared within a function) that were specified without any of the \_\_device\_\_, \_\_shared\_\_ or \_\_constant\_\_ memory space specifiers, if they do not fit into registers or the compiler decides to place them in local memory for other reasons. Automatic variables that the compiler is likely to place in local memory are:

- Arrays for which it cannot determine that they are indexed with constant quantities,

- Large structures or arrays that would consume too much register space,

- Any variable if the kernel uses more registers than available (this is also known as *register spilling*).

The local memory space resides in device memory, so local memory accesses have similar performance characteristics as accesses to global memory. Local memory addresses are translated to global virtual addresses by the AGU unit.

One difference between global and local memory is that local memory is arranged such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address (e.g., same index in an array variable, same member in a structure variable, etc.).

To minimize adverse performance impact of local memory accesses consider the following best practices:

- **Prefer Registers:** For small, thread-private variables that are accessed frequently, aim for them to reside in registers. The compiler often does this automatically for scalar variables.

- **Utilize Shared Memory:** For data that is shared and reused by threads within a block, shared memory is significantly faster than local memory. Explicitly manage data in shared memory to maximize its benefits.

- **Increasing the available register count:** Use a higher limit in [\_\_maxnreg\_\_](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximum-number-of-registers-per-thread) or a lower thread count in [\_\_launch_bounds\_\_](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds) to increase the number of registers available to each thread. This can help reduce register spilling.

- **Increasing the L1 cache size:** If local memory usage is unavoidable, try to cache most local memory accesses in L1 by increasing its size with [cudaFuncSetCacheConfig](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g6699ca1943ac2655effa0d571b2f4f15).

- **Using non-caching global memory loads:** If there are many cache misses in the L1 cache, consider using non-caching global memory loads to reduce contentions by compiling with the nvcc option -Xptxas –def-load-cache=cg (see the [nvcc docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#def-load-cache-dlcm)).

More information on local memory can be found in the [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) and the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#local-memory).

*Shared memory* is located on chip within each Streaming Multiprocessor (SM), so it has much higher bandwidth and much lower latency than either local or global memory. Shared memory can be shared across a compute CTA. Compute CTAs attempting to share data across threads via shared memory must use synchronization operations (such as \_\_syncthreads()) between stores and loads to ensure data written by any one thread is visible to other threads in the CTA. Similarly, threads that need to share data via global memory must use a more heavyweight global memory barrier.

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks that can be accessed simultaneously. Any 32-bit memory read or write request made of 32 addresses that fall in 32 distinct memory banks can therefore be serviced simultaneously, yielding an overall bandwidth that is 32 times as high as the bandwidth of a single request. However, if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized.

A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word (even though the two addresses fall in the same bank). When multiple threads make the same read access, one thread receives the data and then broadcasts it to the other threads. When multiple threads write to the same location, only one thread succeeds in the write; which thread that succeeds is undefined.

Detailed memory metrics are collected by the [Memory Workload Analysis](index.html#sections-and-rules) section.

#### Caches

All GPU units communicate to main memory through the Level 2 cache, also known as the L2. The L2 cache sits between on-chip memory clients and the framebuffer. L2 works in physical-address space. In addition to providing caching functionality, L2 also includes hardware to perform compression and global atomics.



Model of the L2 cache.

The Level 1 Data Cache, or L1, plays a key role in handling global, local, shared, texture, and surface memory reads and writes, as well as reduction and atomic operations. On Volta and Turing architectures there are , there are two L1 caches per TPC, one for each SM. For more information on how L1 fits into the texturing pipeline, see the [TEX unit](index.html#metrics-hw-tex-surf) description. Also note that while this section often uses the name “L1”, it should be understood that the L1 data cache, shared data, and the Texture data cache are one and the same.

L1 receives requests from two units: the SM and TEX. L1 receives global and local memory requests from the SM and receives texture and surface requests from TEX. These operations access memory in the global memory space, which L1 sends through a secondary cache, the L2.

Cache hit and miss rates as well as data transfers are reported in the [Memory Workload Analysis](index.html#sections-and-rules) section.



Model of Load/Store and Texture pipelines for the L1TEX cache.

#### Texture/Surface

The TEX unit performs texture fetching and filtering. Beyond plain texture memory access, TEX is responsible for the addressing, LOD, wrap, filter, and format conversion operations necessary to convert a texture read request into a result.

TEX receives two general categories of requests from the SM via its input interface: texture requests and surface load/store operations. Texture and surface memory space resides in device memory and are cached in L1. Texture and surface memory are allocated as block-linear surfaces (e.g. 2D, 2D Array, 3D). Such surfaces provide a cache-friendly layout of data such that neighboring points on a 2D surface are also located close to each other in memory, which improves access locality. Surface accesses are bounds-checked by the TEX unit prior to accessing memory, which can be used for implementing different texture wrapping modes.

The L1 cache is optimized for 2D spatial locality, so threads of the same warp that read texture or surface addresses that are close together in 2D space will achieve optimal performance. The L1 cache is also designed for streaming fetches with constant latency; a cache hit reduces DRAM bandwidth demand but not fetch latency. Reading device memory through texture or surface memory presents some benefits that can make it an advantageous alternative to reading memory from global or constant memory.

Information on texture and surface memory can be found in the [Memory Workload Analysis](index.html#sections-and-rules) section.

### 2.3.2. Metrics Structure

#### Metrics Overview

NVIDIA Nsight Compute uses an advanced metrics calculation system, designed to help you determine what happened (counters and metrics), and how close the program reached to peak GPU performance (throughputs as a percentage). Every counter has associated peak rates in the database, to allow computing its throughput as a percentage.

Throughput metrics return the maximum percentage value of their constituent counters. These constituents have been carefully selected to represent the sections of the GPU pipeline that govern peak performance. While all counters can be converted to a %-of-peak, not all counters are suitable for peak-performance analysis; examples of unsuitable counters include qualified subsets of activity, and workload residency counters. Using throughput metrics ensures meaningful and actionable analysis.

Two types of peak rates are available for every counter: burst and sustained. Burst rate is the maximum rate reportable in a single clock cycle. Sustained rate is the maximum rate achievable over an infinitely long measurement period, for “typical” operations. For many counters, burst equals sustained. Since the burst rate cannot be exceeded, percentages of burst rate will always be less than 100%. Percentages of sustained rate can occasionally exceed 100% in edge cases.

#### Metrics Entities

While in NVIDIA Nsight Compute, all performance counters are named *metrics*, they can be split further into groups with specific properties. For metrics collected via the *PerfWorks* measurement library, the following entities exist:

**Counters** may be either a raw counter from the GPU, or a calculated counter value. Every counter has four sub-metrics under it, which are also called *roll-ups*:

|        |                                                      |
|--------|------------------------------------------------------|
| `.sum` | The sum of counter values across all unit instances. |
| `.avg` | The average counter value across all unit instances. |
| `.min` | The minimum counter value across all unit instances. |
| `.max` | The maximum counter value across all unit instances. |

Counter roll-ups have the following calculated quantities as built-in sub-metrics:

|  |  |
|----|----|
| `.peak_sustained` | the peak sustained rate |
| `.peak_sustained_active` | the peak sustained rate during unit active cycles |
| `.peak_sustained_active.per_second` | the peak sustained rate during unit active cycles, per second \* |
| `.peak_sustained_elapsed` | the peak sustained rate during unit elapsed cycles |
| `.peak_sustained_elapsed.per_second` | the peak sustained rate during unit elapsed cycles, per second \* |
| `.per_second` | the number of operations per second |
| `.per_cycle_active` | the number of operations per unit active cycle |
| `.per_cycle_elapsed` | the number of operations per unit elapsed cycle |
| `.pct_of_peak_sustained_active` | % of peak sustained rate achieved during unit active cycles |
| `.pct_of_peak_sustained_elapsed` | % of peak sustained rate achieved during unit elapsed cycles |

\* sub-metrics added in NVIDIA Nsight Compute 2022.2.0.

Example: `ncu`` ``--query-metrics-mode`` ``suffix`` ``--metrics`` ``sm__inst_executed`` ``--chip`` ``ga100`

**Ratios** have three sub-metrics:

|             |                                      |
|-------------|--------------------------------------|
| `.pct`      | The value expressed as a percentage. |
| `.ratio`    | The value expressed as a ratio.      |
| `.max_rate` | The ratio’s maximum value.           |

Example: `ncu`` ``--query-metrics-mode`` ``suffix`` ``--metrics`` ``smsp__average_warp_latency`` ``--chip`` ``ga100`

**Throughputs** indicate how close a portion of the GPU reached to peak rate. Every throughput has the following sub-metrics:

|  |  |
|----|----|
| `.pct_of_peak_sustained_active` | % of peak sustained rate achieved during unit active cycles |
| `.pct_of_peak_sustained_elapsed` | % of peak sustained rate achieved during unit elapsed cycles |

Example: `ncu`` ``--query-metrics-mode`` ``suffix`` ``--metrics`` ``sm__throughput`` ``--chip`` ``ga100`

Throughputs have a breakdown of underlying metrics from which the throughput value is computed. You can collect `breakdown:<throughput-metric>` to collect a throughput’s breakdown metrics.

**Deprecated counter sub-metrics:** The following sub-metrics were removed, due to not being useful for performance optimization:

|  |  |
|----|----|
| `.peak_burst` | the peak burst rate |
| `.pct_of_peak_burst_active` | % of peak burst rate achieved during unit active cycles |
| `.pct_of_peak_burst_elapsed` | % of peak burst rate achieved during unit elapsed cycles |
| `.pct_of_peak_burst_region` | % of peak burst rate achieved over a user-specified “range” |
| `.pct_of_peak_burst_frame` | % of peak burst rate achieved over a user-specified “frame” |
| `.pct_of_peak_sustained_region` | % of peak sustained rate achieved over a user-specified “range” time |
| `.pct_of_peak_sustained_frame` | % of peak sustained rate achieved over a user-specified “frame” time |
| `.per_cycle_in_region` | the number of operations per user-specified “range” cycle |
| `.per_cycle_in_frame` | the number of operations per user-specified “frame” cycle |
| `.peak_sustained_region` | the peak sustained rate over a user-specified “range” |
| `.peak_sustained_region.per_second` | the peak sustained rate over a user-specified “range”, per second \* |
| `.peak_sustained_frame` | the peak sustained rate over a user-specified “frame” |
| `.peak_sustained_frame.per_second` | the peak sustained rate over a user-specified “frame”, per second \* |

**Deprecated throughput sub-metrics:** The following sub-metrics were removed, due to not being useful for performance optimization:

|  |  |
|----|----|
| `.pct_of_peak_burst_active` | % of peak burst rate achieved during unit active cycles |
| `.pct_of_peak_burst_elapsed` | % of peak burst rate achieved during unit elapsed cycles |
| `.pct_of_peak_burst_region` | % of peak burst rate achieved over a user-specified “range” time |
| `.pct_of_peak_burst_frame` | % of peak burst rate achieved over a user-specified “frame” time |
| `.pct_of_peak_sustained_region` | % of peak sustained rate achieved over a user-specified “range” |
| `.pct_of_peak_sustained_frame` | % of peak sustained rate achieved over a user-specified “frame” |

In addition to PerfWorks metrics, NVIDIA Nsight Compute uses several other measurement providers that each generate their own metrics. These are explained in the [Metrics Reference](index.html#metrics-reference).

#### Metrics Examples

``` text
## non-metric names -- *not* directly evaluable
sm__inst_executed                   # counter
smsp__average_warp_latency          # ratio
sm__throughput                      # throughput

## a counter's four first-level sub-metrics -- all evaluable
sm__inst_executed.sum
sm__inst_executed.avg
sm__inst_executed.min
sm__inst_executed.max

## all names below are metrics -- all evaluable
l1tex__data_bank_conflicts_pipe_lsu.sum
l1tex__data_bank_conflicts_pipe_lsu.sum.peak_sustained
l1tex__data_bank_conflicts_pipe_lsu.sum.peak_sustained_active
l1tex__data_bank_conflicts_pipe_lsu.sum.peak_sustained_active.per_second
l1tex__data_bank_conflicts_pipe_lsu.sum.peak_sustained_elapsed
l1tex__data_bank_conflicts_pipe_lsu.sum.peak_sustained_elapsed.per_second
l1tex__data_bank_conflicts_pipe_lsu.sum.per_cycle_active
l1tex__data_bank_conflicts_pipe_lsu.sum.per_cycle_elapsed
l1tex__data_bank_conflicts_pipe_lsu.sum.per_second
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_sustained_active
l1tex__data_bank_conflicts_pipe_lsu.sum.pct_of_peak_sustained_elapsed
...
```

#### Metrics Naming Conventions

Counters and metrics *generally* obey the naming scheme:

- Unit-Level Counter : `unit__(subunit?)_(pipestage?)_quantity_(qualifiers?)`

- Interface Counter : `unit__(subunit?)_(pipestage?)_(interface)_quantity_(qualifiers?)`

- Unit Metric : `(counter_name).(rollup_metric)`

- Sub-Metric : `(counter_name).(rollup_metric).(submetric)`

where

- unit: A logical or physical unit of the GPU

- subunit: The subunit within the unit where the counter was measured. Sometimes this is a pipeline mode instead.

- pipestage: The pipeline stage within the subunit where the counter was measured.

- quantity: What is being measured. Generally matches the *dimensional units*.

- qualifiers: Any additional predicates or filters applied to the counter. Often, an unqualified counter can be broken down into several qualified sub-components.

- interface: Of the form `sender2receiver`, where `sender` is the source-unit and `receiver` is the destination-unit.

- rollup_metric: One of sum, avg, min, max.

- submetric: refer to section [Metrics Entities](index.html#metrics-entities)

Components are not always present. Most top-level counters have no qualifiers. Subunit and pipestage may be absent where irrelevant, or there may be many subunit specifiers for detailed counters.

Metrics with `sass_` in the name are SASS metrics collected with SW-patching. They can be collected at different levels:

- unit-level, e.g. [here](index.html#l2-cache-eviction-metrics) or [here](index.html#sass-unit-level-instructions-executed-metrics)

- instruction-level, e.g. [here](index.html#instructions-per-opcode-metrics)

There are more instruction-level SASS metrics which don’t have `sass_` in the name, e.g. [here](index.html#source-metrics). `device__attribute_sass_level` is not a SASS metric.

#### Cycle Metrics

Counters using the term `cycles` in the name report the number of cycles in the unit’s clock domain. Unit-level cycle metrics include:

- `unit__cycles_elapsed` : The number of cycles within a range. The cycles’ DimUnits are specific to the unit’s clock domain.

- `unit__cycles_active` : The number of cycles where the unit was processing data.

- `unit__cycles_stalled` : The number of cycles where the unit was unable to process new data because its output interface was blocked.

- `unit__cycles_idle` : The number of cycles where the unit was idle.

Interface-level cycle counters are often (not always) available in the following variations:

- `unit__(interface)_active` : Cycles where data was transferred from source-unit to destination-unit.

- `unit__(interface)_stalled` : Cycles where the source-unit had data, but the destination-unit was unable to accept data.

#### Instanced Metrics

Metrics collected with NVIDIA Nsight Compute can have a single (aggregate) value, multiple instance values, or both. Instances allow the metric to have multiple sub-values, e.g. representing the value of a source metric at each instruction offset. If a metric has instance values, it often also has a correlation ID for each instance. Correlation IDs and values form a mapping that allows the tool to correlate the values within a context. For source metrics, that context is commonly the address ranges of the functions executed as part of the workload.

You can find which metrics have instance values in the [Metrics Reference](index.html#metrics-reference). In the UI, the [Metric Details](../NsightCompute/index.html#tool-window-metric-details) tool window can be used to conveniently view correlation IDs and instance values for each metric. Also, both the UI and the command line interface provide options to show instance values in addition to a metric aggregate where applicable.

### 2.3.3. Metrics Decoder

The following explains terms found in NVIDIA Nsight Compute metric names, as introduced in [Metrics Structure](index.html#metrics-structure).

### 2.3.4. Units

> [TABLE]
>
> Units {#id41 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.3.5. Subunits

> |  |  |
> |----|----|
> | `aperture_device` | Memory interface to local device memory (dram) |
> | `aperture_peer` | Memory interface to remote device memory |
> | `aperture_sysmem` | Memory interface to system memory |
> | `global` | Global memory is a 49-bit virtual address space that is mapped to physical memory on the device, pinned system memory, or peer memory. Global memory is visible to all threads in the GPU. Global memory is accessed through the SM L1 and GPU L2. |
> | `ilc` | Inline Compressor (ILC) is part of LRC. It compresses writes when possible before data is written to L2 and decompresses data when responding to read requests. |
> | `lg` | Local/Global memory |
> | `local` | Local memory is private storage for an executing thread and is not visible outside of that thread. It is intended for thread-local data like thread stacks and register spills. Local memory has the same latency as global memory. |
> | `lsu` | Load/Store unit |
> | `lsuin` | Load/Store input |
> | `mio` | Memory input/output |
> | `mioc` | Memory input/output control |
> | `shared` | Shared memory is located on chip, so it has much higher bandwidth and much lower latency than either local or global memory. Shared memory can be shared across a compute CTA. |
> | `surface` | Surface memory |
> | `texin` | TEXIN |
> | `texture` | Texture memory |
> | `workid` | ID for a unit of work in a grid that uses cluster launch control (CLC). |
> | `xbar` | The Crossbar (XBAR) is responsible for carrying packets from a given source unit to a specific destination unit. |
>
> Subunits {#id42 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.3.6. Pipelines

Pipelines execute instructions. Some pipelines are physical, i.e. they correspond to physical HW. Other pipelines are logical, i.e. they are an abstraction of one or more physical pipelines. The terms ‘aggregated pipeline’ and ‘sub pipeline’ are used to indicate the hierarchy, but without specifying if it is physical or logical.

Some instructions can execute in one of multiple pipelines, and the decision is dynamic at runtime. Some instructions execute in both of two pipelines, and the association is static. Other instructions execute in only a single pipeline.

> |  |  |
> |----|----|
> | `adu` | Address Divergence Unit. The ADU is responsible for address divergence handling for branches/jumps. It also provides support for constant loads and block-level barrier instructions. |
> | `alu` | Arithmetic Logic Unit. The ALU is responsible for execution of most bit manipulation and logic instructions. It also executes integer instructions, excluding IMAD and IMUL. On NVIDIA Ampere architecture chips, the ALU pipeline performs fast FP32-to-FP16 conversion. ALU is an aggregated pipe composed of ALU Heavy and ALU Lite (part of physical pipe FMA Heavy). |
> | `aluheavy` | Arithmetic Logic Unit Heavy. ALU Heavy is part of the aggregated pipe ALU. |
> | `alulite` | Arithmetic Logic Unit Lite. It is part of the aggregated pipe ALU and a subpipe of the physical pipe FMA Heavy. |
> | `cbu` | Convergence Barrier Unit. The CBU is responsible for warp-level convergence, barrier, and branch instructions. |
> | `fma` | Fused Multiply Add/Accumulate. The FMA pipeline processes most FP32 arithmetic (FADD, FMUL, FMAD). It also performs integer multiplication operations (IMUL, IMAD), as well as integer dot products. On GA10x, FMA is a logical pipeline that indicates peak FP32 and FP16x2 performance. It is an aggregated pipe composed of the FMA Heavy and FMA Lite sub-pipelines. |
> | `fmaheavy` | Fused Multiply Add/Accumulate Heavy. FMA Heavy performs FP32 arithmetic (FADD, FMUL, FMAD), FP16 arithmetic (HADD2, HMUL2, HFMA2), integer multiplication operations (IMUL, IMAD), and integer dot products. Shared FMA Heavy is an aggregated, physical pipe composed of the subpipes FMA Heavy and ALU Lite. Its FMA Heavy subpipe can execute instructions from FMA. Its ALU Lite subpipe can execute instructions from ALU. |
> | `fmalite` | Fused Multiply Add/Accumulate Lite. FMA Lite performs FP32 arithmetic (FADD, FMUL, FMA) and FP16 arithmetic (HADD2, HMUL2, HFMA2). It is part of the aggregated pipe FMA. |
> | `fp16` | Half-precision floating-point. On Volta, Turing and NVIDIA GA100, the FP16 pipeline performs paired FP16 instructions (FP16x2). It also contains a fast FP32-to-FP16 and FP16-to-FP32 converter. Starting with GA10x chips, this functionality is part of the FMA pipeline. |
> | `fp64` | Double-precision floating-point. The implementation of FP64 varies greatly per chip. |
> | `lsu` | Load Store Unit. The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for global, local, and shared memory. It also issues special register reads (S2R), shuffles, and CTA-level arrive/wait barrier instructions to the L1TEX unit. |
> | `tc` | Tensor Core. The TC pipeline executes UTCBAR, UTCCP, UTC\*MMA, UTCSHIFT and UTC\*SWS instructions. It is different from the Tensor pipeline. |
> | `tensor` | The Tensor pipeline executes various MMA instructions. It is different from the Tensor Core pipeline. |
> | `tex` | Texture Unit. The SM texture pipeline forwards texture and surface instructions to the L1TEX unit’s TEXIN stage. On GPUs where FP64 or Tensor pipelines are decoupled, the texture pipeline forwards those types of instructions, too. |
> | `tma` | Tensor Memory Accelerator. (Tensor Memory Access Unit) Provides efficient data transfer mechanisms between global and shared memories with the ability to understand and traverse multidimensional data layouts. |
> | `tmem` | Tensor Memory. The TMEM pipeline executes FENCE.VIEW.ASYNC.T, LDT(M) and STT(M) instructions. TMEM also refers to the dedicated tensor memory within the SM sub-partition (SMSP). |
> | `uniform` | Uniform Data Path. This scalar unit executes instructions where all threads use the same input and generate the same output. |
> | `xu` | Transcendental and Data Type Conversion Unit. The XU pipeline is responsible for special functions such as sin, cos, and reciprocal square root. It is also responsible for int-to-float, and float-to-int type conversions. |
>
> Pipelines {#id43 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.3.7. Quantities

> |  |  |
> |----|----|
> | `instruction` | An assembly (SASS) instruction. Each executed instruction may generate zero or more requests. |
> | `request` | A command into a HW unit to perform some action, e.g. load data from some memory location. Each request accesses one or more sectors. |
> | `sector` | Aligned 32 byte-chunk of memory in a cache line or device memory. An L1 or L2 cache line is four sectors, i.e. 128 bytes. Sector accesses are classified as hits if the tag is present and the sector-data is present within the cache line. Tag-misses and tag-hit-data-misses are all classified as misses. |
> | `tag` | Unique key to a cache line. A request may look up multiple tags, if the thread addresses do not all fall within a single cache line-aligned region. The L1 and L2 both have 128 byte cache lines. Tag accesses may be classified as hits or misses. |
> | `wavefront` | Unique “work package” generated at the end of the processing stage for requests. All work items of a wavefront are processed in parallel, while work items of different wavefronts are serialized and processed on different cycles. At least one wavefront is generated for each request. |
>
> Quantities {#id44 .colwidths-auto .table-no-stripes .docutils .align-default}

A simplified model for the processing in L1TEX for Volta and newer architectures can be described as follows: When an SM executes a global or local memory instruction for a warp, a single *request* is sent to L1TEX. This request communicates the information for all participating threads of this warp (up to 32). For local and global memory, based on the access pattern and the participating threads, the request requires to access a number of cache lines, and *sectors* within these cache lines. The L1TEX unit has internally multiple processing stages operating in a pipeline.

A *wavefront* is the maximum unit that can pass through that pipeline stage per cycle. If not all cache lines or sectors can be accessed in a single wavefront, multiple wavefronts are created and sent for processing one by one, i.e. in a serialized manner. Limitations of the work within a wavefront may include the need for a consistent memory space, a maximum number of cache lines that can be accessed, as well as various other reasons. Each wavefront then flows through the L1TEX pipeline and fetches the sectors handled in that wavefront. The given relationships of the three key values in this model are *requests:sectors is 1:N, wavefronts:sectors 1:N, and requests:wavefronts is 1:N*.

A wavefront is described as a (work) package that can be processed at once, i.e. there is a notion of processing one wavefront per cycle in L1TEX. Wavefronts therefore represent the number of cycles required to process the requests, while the number of sectors per request is a property of the *access pattern* of the memory instruction for all participating threads. For example, it is possible to have a memory instruction that requires 4 sectors per request in 1 wavefront. However, you can also have a memory instruction having 4 sectors per request, but requiring 2 or more wavefronts.

### 2.3.8. Range and Precision

#### Overview

In general, measurement values that lie outside the expected logical range of a metric can be attributed to one or more of the below root-causes. If values are exceeding such range, they are not clamped by the tool to their expected value on purpose to ensure that the rest of the profiler report remains self-consistent.

#### Asynchronous GPU activity

GPU engines other than the one measured by a metric (display, copy engine, video encoder, video decoder, etc.) potentially access shared resources during profiling. Such chip-global shared resources include L2, DRAM, PCIe, and NVLINK. If the kernel launch is small, the other engine(s) can cause significant confusion in e.g. the DRAM results, since it is not possible to isolate the DRAM traffic of the SM. To reduce the impact of such asynchronous units, consider profiling on a GPU without active display and without other processes that can access the GPU at the time.

#### Multi-pass data collection

Out-of-range metrics often occur when the profiler [replays](index.html#kernel-replay) the kernel launch to collect metrics, and work distribution is significantly different across replay passes. A metric such as hit rate (hits / queries) can have significant error if hits and queries are collected on different passes and the kernel does not saturate the GPU to reach a steady state (generally \> 20 µs). Similarly, it can show unexpected values when the workload is inherently variable, as e.g. in the case of spin loops.

To mitigate the issue, when applicable try to increase the measured workload to allow the GPU to reach a steady state for each launch. Reducing the number of metrics collected at the same time can also improve precision by increasing the likelihood that counters contributing to one metric are collected in a single pass.

#### Workload Durations

Nsight Compute measures the duration of workloads (e.g., CUDA kernels) differently than Nsight Systems (via CUPTI). This can lead to discrepancies in the reported duration for the same workload between the two tools.

The following factors impact the measured duration:

- [Clock control](index.html#clock-control): Nsight Compute locks SM clocks [by default](../NsightComputeCli/index.html#profile). Nsight Systems does not lock clocks. When comparing results, we recommend to lock clocks with `nvidia-smi` externally before profiling and use `--clock-control`` ``none` for ncu.

- [Cache control](index.html#cache-control): Nsight Compute flushes all GPU caches [by default](../NsightComputeCli/index.html#profile) between replay passes. Nsight Systems collects data only in a single pass, and therefore also does not flush any caches. If the workload is highly sensitive to cache state, it’s recommended to use `--replay-mode`` ``application`` ``--cache-control`` ``none` for ncu to let the application handle priming the caches implicitly.

- [Serialization](index.html#serialization): Nsight Compute serializes kernel launches, unless a dedicated replay mode is used. Nsight Systems does not serialize launches. Concurrent kernels on the same device can lead to differences in the reported duration between the two tools.

- [Tool overhead](index.html#overhead): Nsight Compute introduces varying overhead around and during the execution of a workload for the purpose of metric collection:

  - Metrics collected using software-patching introduce significant overhead during the workload execution. For this reason, the workload duration is never collected in such a replay pass.

  - Preparing the GPU for metric collection and processing the profiled data adds overhead before and after each replay pass. This overhead is not included in the reported duration.

  - It is important though to note that this makes it impossible to derive the workload duration using host-side timers or CUDA events when profiling with Nsight Compute, as such methods would include the overhead introduced by the tool.

- **Duration measurement**: Nsight Compute and Nsight Systems/CUPTI use different methods to measure the duration of a workload (`gpu__time_duration` metric in Nsight Compute):

  - For the start timestamp, Nsight Compute inserts a *WaitForIdle* and a semaphore timestamp after the launch setup (copy parameters, copy qmd) and before the submission of the command to execute the grid. These semaphores are processed by the GPU’s front-end and the timestamp is taken when the semaphore command is executed. This timestamp consequently occurs before the grid’s blocks and warps are scheduled and distributed to the SMs.

    Before Blackwell, Nsight Systems measures the start timestamp by replacing the entry point of the kernel with a prolog that all warps execute. Logical thread 0 writes out the timestamp. This method is required to trace concurrent kernels. From Blackwell onwards, Nsight Systems collects the start timestamp using a HW method. It is taken before the first CTA is launched on the SM.

  - For the end timestamp, Nsight Compute inserts a *WaitForIdle* followed by another semaphore into the command buffer.

    Before Blackwell, Nsight Systems instead uses a semaphore that instructs the scheduler unit to collect the timestamp upon completion of all blocks of the grid. This can occur before or after the semaphore method used by Nsight Compute. From Blackwell onwards, Nsight Systems collects the end timestamp using a HW method. It is taken after the final memory barrier is issued.

  - In general, Nsight Systems should report shorter durations since the start timestamp is as close as possible (later, never earlier) to the first instruction executed by the kernel on a SM.

  - The above method is used in Nsight Compute to be compatible with other ways to measure “duration” in the profiler, such as *elapsed cycles* metrics.

  - Note that for *Workload Execution* rows in the PM Sampling [timeline](../NsightCompute/index.html#timeline), Nsight Compute and Nsight Systems use the same method to measure start and end timestamps.

#### Tool issue

If you still observe metric issues after following the guidelines above, please [reach out to us](https://forums.developer.nvidia.com/c/developer-tools/nsight-compute) and describe your issue.

## 2.4. Metrics Reference

### 2.4.1. Overview

Most metrics in NVIDIA Nsight Compute can be queried using the ncu command line interface’s [–query-metrics](../NsightComputeCli/index.html#command-line-options-profile) option.

The following metrics can be collected explicitly but do not follow the naming scheme explained in [Metrics Structure](index.html#metrics-structure). They should be used as-is instead. These metrics can be listed using `--query-metrics` as per [–query-metrics-collection](../NsightComputeCli/index.html#command-line-options-profile) option.

`launch__*` metrics are collected per kernel launch, and do not require an additional replay pass. They are available as part of the kernel launch parameters (such as grid size, block size, …) or are computed using the [CUDA Occupancy Calculator](../NsightCompute/index.html#occupancy-calculator). For range-based results, `launch__*` metrics are instanced for each launch, with the exception of `launch__uses_*`, `launch__graph_*` and `launch__occupancy_per_*`.

### 2.4.2. Launch Metrics

> [TABLE]
>
> Launch Metrics {#id45 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.3. Occupancy Metrics

> |  |  |
> |----|----|
> | `sm__maximum_warps_avg_per_active_cycle` | Theoretical Active Warps Per SM |
> | `sm__maximum_warps_per_active_cycle_pct` | Theoretical Occupancy |
> | `smsp__maximum_warps_avg_per_active_cycle` | Theoretical Warps Per Scheduler |
>
> Occupancy Metrics {#id46 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.4. NVLink Topology Metrics

> [TABLE]
>
> NVLink Topology Metrics {#id47 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.5. NUMA Topology Metrics

> [TABLE]
>
> NUMA Topology Metrics {#id48 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.6. Device Attributes

`device__attribute_*` metrics represent [CUDA device attributes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g49e2f8c2c0bd6fe264f2fc970912e5cd). Collecting them does not require an additional kernel replay pass, as their value is available from the CUDA driver for each CUDA device.

See below for custom `device__attribute_*` metrics.

> |  |  |
> |----|----|
> | `device__attribute_architecture` | Chip architecture of the CUDA device. |
> | `device__attribute_confidential_computing_mode` | Confidential computing mode. |
> | `device__attribute_device_index` | Device index. |
> | `device__attribute_display_name` | Product name of the CUDA device. |
> | `device__attribute_export_control_chip` | Indicates if the device is an export control chip. |
> | `device__attribute_fb_bus_width` | Frame buffer bus width. |
> | `device__attribute_fbp_count` | Total number of frame buffer partitions. |
> | `device__attribute_implementation` | Chip implementation of the CUDA device. |
> | `device__attribute_l2s_count` | Total number of Level 2 cache slices. |
> | `device__attribute_limits_max_cta_per_sm` | Maximum number of CTA per SM. |
> | `device__attribute_max_gpu_frequency_khz` | Maximum GPU frequency in kilohertz. |
> | `device__attribute_max_ipc_per_multiprocessor` | Maximum number of instructions per clock per multiprocessor. |
> | `device__attribute_max_ipc_per_scheduler` | Maximum number of instructions per clock per scheduler. |
> | `device__attribute_max_mem_frequency_khz` | Peak memory frequency in kilohertz. |
> | `device__attribute_max_registers_per_thread` | Maximum number of registers available per thread. |
> | `device__attribute_max_warps_per_multiprocessor` | Maximum number of warps per multiprocessor. |
> | `device__attribute_max_warps_per_scheduler` | Maximum number of warps per scheduler. |
> | `device__attribute_num_l2s_per_fbp` | Number of Level 2 cache slices per frame buffer partition. |
> | `device__attribute_num_schedulers_per_multiprocessor` | Number of schedulers per multiprocessor. |
> | `device__attribute_num_tex_per_multiprocessor` | Number of TEX unit per multiprocessor. |
> | `device__attribute_sass_level` | SASS level. |

### 2.4.7. Warp Stall Reasons

Collected using warp scheduler state sampling. They are incremented regardless if the scheduler issued an instruction in the same cycle or not. These metrics have instance values mapping from the function address (uint64) to the number of samples (uint64).

> |  |  |
> |----|----|
> | `smsp__pcsamp_warps_issue_stalled_barrier` | Warp was stalled waiting for sibling warps at a CTA barrier. A high number of warps waiting at a barrier is commonly caused by diverging code paths before a barrier. This causes some warps to wait a long time until other warps reach the synchronization point. Whenever possible, try to divide up the work into blocks of uniform workloads. If the block size is 512 threads or greater, consider splitting it into smaller groups. This can increase eligible warps without affecting occupancy, unless shared memory becomes a new occupancy limiter. Also, try to identify which barrier instruction causes the most stalls, and optimize the code executed before that synchronization point first. |
> | `smsp__pcsamp_warps_issue_stalled_branch_resolving` | Warp was stalled waiting for a branch target to be computed, and the warp program counter to be updated. To reduce the number of stalled cycles, consider using fewer jump/branch operations and reduce control flow divergence, e.g. by reducing or coalescing conditionals in your code. See also the related No Instructions state. |
> | `smsp__pcsamp_warps_issue_stalled_dispatch_stall` | Warp was stalled waiting on a dispatch stall. A warp stalled during dispatch has an instruction ready to issue, but the dispatcher holds back issuing the warp due to other conflicts or events. |
> | `smsp__pcsamp_warps_issue_stalled_drain` | Warp was stalled after EXIT waiting for all outstanding memory operations to complete so that warp’s resources can be freed. A high number of stalls due to draining warps typically occurs when a lot of data is written to memory towards the end of a kernel. Make sure the memory access patterns of these store operations are optimal for the target architecture and consider parallelized data reduction, if applicable. |
> | `smsp__pcsamp_warps_issue_stalled_imc_miss` | Warp was stalled waiting for an immediate constant cache (IMC) miss. A read from constant memory costs one memory read from device memory only on a cache miss; otherwise, it just costs one read from the constant cache. Immediate constants are encoded into the SASS instruction as ‘c\[bank\]\[offset\]’. Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly with the number of unique addresses read by all threads within a warp. As such, the constant cache is best when threads in the same warp access only a few distinct locations. If all threads of a warp access the same location, then constant memory can be as fast as a register access. |
> | `smsp__pcsamp_warps_issue_stalled_lg_throttle` | Warp was stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. Typically, this stall occurs only when executing local or global memory instructions extremely frequently. Avoid redundant global memory accesses. Try to avoid using thread-local memory by checking if dynamically indexed arrays are declared in local scope, or if the kernel has excessive register pressure causing spills. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions. |
> | `smsp__pcsamp_warps_issue_stalled_long_scoreboard` | Warp was stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently used data to shared memory. |
> | `smsp__pcsamp_warps_issue_stalled_math_pipe_throttle` | Warp was stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. |
> | `smsp__pcsamp_warps_issue_stalled_membar` | Warp was stalled waiting on a memory barrier. Avoid executing any unnecessary memory barriers and assure that any outstanding memory operations are fully optimized for the target architecture. |
> | `smsp__pcsamp_warps_issue_stalled_mio_throttle` | Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure. |
> | `smsp__pcsamp_warps_issue_stalled_misc` | Warp was stalled for a miscellaneous hardware reason. |
> | `smsp__pcsamp_warps_issue_stalled_no_instructions` | Warp was stalled waiting to be selected to fetch an instruction or waiting on an instruction cache miss. A high number of warps not having an instruction fetched is typical for very short kernels with less than one full wave of work in the grid. Excessively jumping across large blocks of assembly code can also lead to more warps stalled for this reason, if this causes misses in the instruction cache. See also the related Branch Resolving state. |
> | `smsp__pcsamp_warps_issue_stalled_not_selected` | Warp was stalled waiting for the micro scheduler to select the warp to issue. Not selected warps are eligible warps that were not picked by the scheduler to issue that cycle as another warp was selected. A high number of not selected warps typically means you have sufficient warps to cover warp latencies and you may consider reducing the number of active warps to possibly increase cache coherence and data locality. |
> | `smsp__pcsamp_warps_issue_stalled_selected` | Warp was selected by the micro scheduler and issued an instruction. |
> | `smsp__pcsamp_warps_issue_stalled_short_scoreboard` | Warp was stalled waiting for a scoreboard dependency on a MIO (memory input/output) operation (not to L1TEX). The primary reason for a high number of stalls due to short scoreboards is typically memory operations to shared memory. Other reasons include frequent execution of special math instructions (e.g. MUFU) or dynamic branching (e.g. BRX, JMX). Consult the Memory Workload Analysis section to verify if there are shared memory operations and reduce bank conflicts, if reported. Assigning frequently accessed values to variables can assist the compiler in using low-latency registers instead of direct memory accesses. |
> | `smsp__pcsamp_warps_issue_stalled_sleeping` | Warp was stalled due to all threads in the warp being in the blocked, yielded, or sleep state. Reduce the number of executed NANOSLEEP instructions, lower the specified time delay, and attempt to group threads in a way that multiple threads in a warp sleep at the same time. |
> | `smsp__pcsamp_warps_issue_stalled_tex_throttle` | Warp was stalled waiting for the L1 instruction queue for texture operations to be not full. This stall reason is high in cases of extreme utilization of the L1TEX pipeline. Try issuing fewer texture fetches, surface loads, surface stores, or decoupled math operations. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions. Consider converting texture lookups or surface loads into global memory lookups. Texture can accept four threads’ requests per cycle, whereas global accepts 32 threads. |
> | `smsp__pcsamp_warps_issue_stalled_wait` | Warp was stalled waiting on a fixed latency execution dependency. Typically, this stall reason should be very low and only shows up as a top contributor in already highly optimized kernels. Try to hide the corresponding instruction latencies by increasing the number of active warps, restructuring the code or unrolling loops. Furthermore, consider switching to lower-latency instructions, e.g. by making use of fast math compiler options. |
> | `smsp__pcsamp_warps_issue_stalled_warpgroup_arrive` | Warp was stalled waiting on a WARPGROUP.ARRIVES or WARPGROUP.WAIT instruction. |
>
> Warp Stall Reasons {#id49 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.8. Warp Stall Reasons (Not Issued)

Collected using warp scheduler state sampling. They are incremented only on cycles in which the warp scheduler issued no instruction. These metrics have instance values mapping from the function address (uint64) to the number of samples (uint64).

> |  |  |
> |----|----|
> | `smsp__pcsamp_warps_issue_stalled_barrier_not_issued` | Warp was stalled waiting for sibling warps at a CTA barrier. A high number of warps waiting at a barrier is commonly caused by diverging code paths before a barrier. This causes some warps to wait a long time until other warps reach the synchronization point. Whenever possible, try to divide up the work into blocks of uniform workloads. If the block size is 512 threads or greater, consider splitting it into smaller groups. This can increase eligible warps without affecting occupancy, unless shared memory becomes a new occupancy limiter. Also, try to identify which barrier instruction causes the most stalls, and optimize the code executed before that synchronization point first. |
> | `smsp__pcsamp_warps_issue_stalled_branch_resolving_not_issued` | Warp was stalled waiting for a branch target to be computed, and the warp program counter to be updated. To reduce the number of stalled cycles, consider using fewer jump/branch operations and reduce control flow divergence, e.g. by reducing or coalescing conditionals in your code. See also the related No Instructions state. |
> | `smsp__pcsamp_warps_issue_stalled_dispatch_stall_not_issued` | Warp was stalled waiting on a dispatch stall. A warp stalled during dispatch has an instruction ready to issue, but the dispatcher holds back issuing the warp due to other conflicts or events. |
> | `smsp__pcsamp_warps_issue_stalled_drain_not_issued` | Warp was stalled after EXIT waiting for all memory operations to complete so that warp resources can be freed. A high number of stalls due to draining warps typically occurs when a lot of data is written to memory towards the end of a kernel. Make sure the memory access patterns of these store operations are optimal for the target architecture and consider parallelized data reduction, if applicable. |
> | `smsp__pcsamp_warps_issue_stalled_imc_miss_not_issued` | Warp was stalled waiting for an immediate constant cache (IMC) miss. A read from constant memory costs one memory read from device memory only on a cache miss; otherwise, it just costs one read from the constant cache. Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly with the number of unique addresses read by all threads within a warp. As such, the constant cache is best when threads in the same warp access only a few distinct locations. If all threads of a warp access the same location, then constant memory can be as fast as a register access. |
> | `smsp__pcsamp_warps_issue_stalled_lg_throttle_not_issued` | Warp was stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. Typically, this stall occurs only when executing local or global memory instructions extremely frequently. Avoid redundant global memory accesses. Try to avoid using thread-local memory by checking if dynamically indexed arrays are declared in local scope, or if the kernel has excessive register pressure causing spills. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions. |
> | `smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued` | Warp was stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently used data to shared memory. |
> | `smsp__pcsamp_warps_issue_stalled_math_pipe_throttle_not_issued` | Warp was stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. |
> | `smsp__pcsamp_warps_issue_stalled_membar_not_issued` | Warp was stalled waiting on a memory barrier. Avoid executing any unnecessary memory barriers and assure that any outstanding memory operations are fully optimized for the target architecture. |
> | `smsp__pcsamp_warps_issue_stalled_mio_throttle_not_issued` | Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure. |
> | `smsp__pcsamp_warps_issue_stalled_misc_not_issued` | Warp was stalled for a miscellaneous hardware reason. |
> | `smsp__pcsamp_warps_issue_stalled_no_instructions_not_issued` | Warp was stalled waiting to be selected to fetch an instruction or waiting on an instruction cache miss. A high number of warps not having an instruction fetched is typical for very short kernels with less than one full wave of work in the grid. Excessively jumping across large blocks of assembly code can also lead to more warps stalled for this reason, if this causes misses in the instruction cache. See also the related Branch Resolving state. |
> | `smsp__pcsamp_warps_issue_stalled_not_selected_not_issued` | Warp was stalled waiting for the micro scheduler to select the warp to issue. Not selected warps are eligible warps that were not picked by the scheduler to issue that cycle as another warp was selected. A high number of not selected warps typically means you have sufficient warps to cover warp latencies and you may consider reducing the number of active warps to possibly increase cache coherence and data locality. |
> | `smsp__pcsamp_warps_issue_stalled_selected_not_issued` | Warp was selected by the micro scheduler and issued an instruction. |
> | `smsp__pcsamp_warps_issue_stalled_short_scoreboard_not_issued` | Warp was stalled waiting for a scoreboard dependency on a MIO (memory input/output) operation (not to L1TEX). The primary reason for a high number of stalls due to short scoreboards is typically memory operations to shared memory. Other reasons include frequent execution of special math instructions (e.g. MUFU) or dynamic branching (e.g. BRX, JMX). Consult the Memory Workload Analysis section to verify if there are shared memory operations and reduce bank conflicts, if reported. Assigning frequently accessed values to variables can assist the compiler in using low-latency registers instead of direct memory accesses. |
> | `smsp__pcsamp_warps_issue_stalled_sleeping_not_issued` | Warp was stalled due to all threads in the warp being in the blocked, yielded, or sleep state. Reduce the number of executed NANOSLEEP instructions, lower the specified time delay, and attempt to group threads in a way that multiple threads in a warp sleep at the same time. |
> | `smsp__pcsamp_warps_issue_stalled_tex_throttle_not_issued` | Warp was stalled waiting for the L1 instruction queue for texture operations to be not full. This stall reason is high in cases of extreme utilization of the L1TEX pipeline. Try issuing fewer texture fetches, surface loads, surface stores, or decoupled math operations. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions. Consider converting texture lookups or surface loads into global memory lookups. Texture can accept four threads’ requests per cycle, whereas global accepts 32 threads. |
> | `smsp__pcsamp_warps_issue_stalled_wait_not_issued` | Warp was stalled waiting on a fixed latency execution dependency. Typically, this stall reason should be very low and only shows up as a top contributor in already highly optimized kernels. Try to hide the corresponding instruction latencies by increasing the number of active warps, restructuring the code or unrolling loops. Furthermore, consider switching to lower-latency instructions, e.g. by making use of fast math compiler options. |
> | `smsp__pcsamp_warps_issue_stalled_warpgroup_arrive_not_issued` | Warp was stalled waiting on a WARPGROUP.ARRIVES or WARPGROUP.WAIT instruction. |
>
> Warp Stall Reasons (Not Issued) {#id50 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.9. Warp Stalls per Warp ID

Collected using warp scheduler state sampling. They are incremented regardless if the scheduler issued an instruction in the same cycle or not. These metrics have instance values mapping from \<SMSP ID\>:\<Warp ID\> (string) to the number of samples (uint64).

> |  |  |
> |----|----|
> | `smsp__warpidsamp_warps_issue_stalled_barrier` | Warp was stalled waiting for sibling warps at a CTA barrier. A high number of warps waiting at a barrier is commonly caused by diverging code paths before a barrier. This causes some warps to wait a long time until other warps reach the synchronization point. Whenever possible, try to divide up the work into blocks of uniform workloads. If the block size is 512 threads or greater, consider splitting it into smaller groups. This can increase eligible warps without affecting occupancy, unless shared memory becomes a new occupancy limiter. Also, try to identify which barrier instruction causes the most stalls, and optimize the code executed before that synchronization point first. |
> | `smsp__warpidsamp_warps_issue_stalled_branch_resolving` | Warp was stalled waiting for a branch target to be computed, and the warp program counter to be updated. To reduce the number of stalled cycles, consider using fewer jump/branch operations and reduce control flow divergence, e.g. by reducing or coalescing conditionals in your code. See also the related No Instructions state. |
> | `smsp__warpidsamp_warps_issue_stalled_dispatch_stall` | Warp was stalled waiting on a dispatch stall. A warp stalled during dispatch has an instruction ready to issue, but the dispatcher holds back issuing the warp due to other conflicts or events. |
> | `smsp__warpidsamp_warps_issue_stalled_drain` | Warp was stalled after EXIT waiting for all outstanding memory operations to complete so that warp’s resources can be freed. A high number of stalls due to draining warps typically occurs when a lot of data is written to memory towards the end of a kernel. Make sure the memory access patterns of these store operations are optimal for the target architecture and consider parallelized data reduction, if applicable. |
> | `smsp__warpidsamp_warps_issue_stalled_imc_miss` | Warp was stalled waiting for an immediate constant cache (IMC) miss. A read from constant memory costs one memory read from device memory only on a cache miss; otherwise, it just costs one read from the constant cache. Immediate constants are encoded into the SASS instruction as ‘c\[bank\]\[offset\]’. Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly with the number of unique addresses read by all threads within a warp. As such, the constant cache is best when threads in the same warp access only a few distinct locations. If all threads of a warp access the same location, then constant memory can be as fast as a register access. |
> | `smsp__warpidsamp_warps_issue_stalled_lg_throttle` | Warp was stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. Typically, this stall occurs only when executing local or global memory instructions extremely frequently. Avoid redundant global memory accesses. Try to avoid using thread-local memory by checking if dynamically indexed arrays are declared in local scope, or if the kernel has excessive register pressure causing spills. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions. |
> | `smsp__warpidsamp_warps_issue_stalled_long_scoreboard` | Warp was stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently used data to shared memory. |
> | `smsp__warpidsamp_warps_issue_stalled_math_pipe_throttle` | Warp was stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. |
> | `smsp__warpidsamp_warps_issue_stalled_membar` | Warp was stalled waiting on a memory barrier. Avoid executing any unnecessary memory barriers and assure that any outstanding memory operations are fully optimized for the target architecture. |
> | `smsp__warpidsamp_warps_issue_stalled_mio_throttle` | Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure. |
> | `smsp__warpidsamp_warps_issue_stalled_misc` | Warp was stalled for a miscellaneous hardware reason. |
> | `smsp__warpidsamp_warps_issue_stalled_no_instructions` | Warp was stalled waiting to be selected to fetch an instruction or waiting on an instruction cache miss. A high number of warps not having an instruction fetched is typical for very short kernels with less than one full wave of work in the grid. Excessively jumping across large blocks of assembly code can also lead to more warps stalled for this reason, if this causes misses in the instruction cache. See also the related Branch Resolving state. |
> | `smsp__warpidsamp_warps_issue_stalled_not_selected` | Warp was stalled waiting for the micro scheduler to select the warp to issue. Not selected warps are eligible warps that were not picked by the scheduler to issue that cycle as another warp was selected. A high number of not selected warps typically means you have sufficient warps to cover warp latencies and you may consider reducing the number of active warps to possibly increase cache coherence and data locality. |
> | `smsp__warpidsamp_warps_issue_stalled_selected` | Warp was selected by the micro scheduler and issued an instruction. |
> | `smsp__warpidsamp_warps_issue_stalled_short_scoreboard` | Warp was stalled waiting for a scoreboard dependency on a MIO (memory input/output) operation (not to L1TEX). The primary reason for a high number of stalls due to short scoreboards is typically memory operations to shared memory. Other reasons include frequent execution of special math instructions (e.g. MUFU) or dynamic branching (e.g. BRX, JMX). Consult the Memory Workload Analysis section to verify if there are shared memory operations and reduce bank conflicts, if reported. Assigning frequently accessed values to variables can assist the compiler in using low-latency registers instead of direct memory accesses. |
> | `smsp__warpidsamp_warps_issue_stalled_sleeping` | Warp was stalled due to all threads in the warp being in the blocked, yielded, or sleep state. Reduce the number of executed NANOSLEEP instructions, lower the specified time delay, and attempt to group threads in a way that multiple threads in a warp sleep at the same time. |
> | `smsp__warpidsamp_warps_issue_stalled_tex_throttle` | Warp was stalled waiting for the L1 instruction queue for texture operations to be not full. This stall reason is high in cases of extreme utilization of the L1TEX pipeline. Try issuing fewer texture fetches, surface loads, surface stores, or decoupled math operations. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions. Consider converting texture lookups or surface loads into global memory lookups. Texture can accept four threads’ requests per cycle, whereas global accepts 32 threads. |
> | `smsp__warpidsamp_warps_issue_stalled_wait` | Warp was stalled waiting on a fixed latency execution dependency. Typically, this stall reason should be very low and only shows up as a top contributor in already highly optimized kernels. Try to hide the corresponding instruction latencies by increasing the number of active warps, restructuring the code or unrolling loops. Furthermore, consider switching to lower-latency instructions, e.g. by making use of fast math compiler options. |
> | `smsp__warpidsamp_warps_issue_stalled_warpgroup_arrive` | Warp was stalled waiting on a WARPGROUP.ARRIVES or WARPGROUP.WAIT instruction. |
>
> Warp Stalls per Warp ID {#id51 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.10. Warp Stalls per Warp ID (Not Issued)

Collected using warp scheduler state sampling. They are incremented only on cycles in which the warp scheduler issued no instruction. These metrics have instance values mapping from \<SMSP ID\>:\<Warp ID\> (string) to the number of samples (uint64).

> |  |  |
> |----|----|
> | `smsp__warpidsamp_warps_issue_stalled_barrier_not_issued` | Warp was stalled waiting for sibling warps at a CTA barrier. A high number of warps waiting at a barrier is commonly caused by diverging code paths before a barrier. This causes some warps to wait a long time until other warps reach the synchronization point. Whenever possible, try to divide up the work into blocks of uniform workloads. If the block size is 512 threads or greater, consider splitting it into smaller groups. This can increase eligible warps without affecting occupancy, unless shared memory becomes a new occupancy limiter. Also, try to identify which barrier instruction causes the most stalls, and optimize the code executed before that synchronization point first. |
> | `smsp__warpidsamp_warps_issue_stalled_branch_resolving_not_issued` | Warp was stalled waiting for a branch target to be computed, and the warp program counter to be updated. To reduce the number of stalled cycles, consider using fewer jump/branch operations and reduce control flow divergence, e.g. by reducing or coalescing conditionals in your code. See also the related No Instructions state. |
> | `smsp__warpidsamp_warps_issue_stalled_dispatch_stall_not_issued` | Warp was stalled waiting on a dispatch stall. A warp stalled during dispatch has an instruction ready to issue, but the dispatcher holds back issuing the warp due to other conflicts or events. |
> | `smsp__warpidsamp_warps_issue_stalled_drain_not_issued` | Warp was stalled after EXIT waiting for all memory operations to complete so that warp resources can be freed. A high number of stalls due to draining warps typically occurs when a lot of data is written to memory towards the end of a kernel. Make sure the memory access patterns of these store operations are optimal for the target architecture and consider parallelized data reduction, if applicable. |
> | `smsp__warpidsamp_warps_issue_stalled_imc_miss_not_issued` | Warp was stalled waiting for an immediate constant cache (IMC) miss. A read from constant memory costs one memory read from device memory only on a cache miss; otherwise, it just costs one read from the constant cache. Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly with the number of unique addresses read by all threads within a warp. As such, the constant cache is best when threads in the same warp access only a few distinct locations. If all threads of a warp access the same location, then constant memory can be as fast as a register access. |
> | `smsp__warpidsamp_warps_issue_stalled_lg_throttle_not_issued` | Warp was stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. Typically, this stall occurs only when executing local or global memory instructions extremely frequently. Avoid redundant global memory accesses. Try to avoid using thread-local memory by checking if dynamically indexed arrays are declared in local scope, or if the kernel has excessive register pressure causing spills. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions. |
> | `smsp__warpidsamp_warps_issue_stalled_long_scoreboard_not_issued` | Warp was stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently used data to shared memory. |
> | `smsp__warpidsamp_warps_issue_stalled_math_pipe_throttle_not_issued` | Warp was stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. |
> | `smsp__warpidsamp_warps_issue_stalled_membar_not_issued` | Warp was stalled waiting on a memory barrier. Avoid executing any unnecessary memory barriers and assure that any outstanding memory operations are fully optimized for the target architecture. |
> | `smsp__warpidsamp_warps_issue_stalled_mio_throttle_not_issued` | Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure. |
> | `smsp__warpidsamp_warps_issue_stalled_misc_not_issued` | Warp was stalled for a miscellaneous hardware reason. |
> | `smsp__warpidsamp_warps_issue_stalled_no_instructions_not_issued` | Warp was stalled waiting to be selected to fetch an instruction or waiting on an instruction cache miss. A high number of warps not having an instruction fetched is typical for very short kernels with less than one full wave of work in the grid. Excessively jumping across large blocks of assembly code can also lead to more warps stalled for this reason, if this causes misses in the instruction cache. See also the related Branch Resolving state. |
> | `smsp__warpidsamp_warps_issue_stalled_not_selected_not_issued` | Warp was stalled waiting for the micro scheduler to select the warp to issue. Not selected warps are eligible warps that were not picked by the scheduler to issue that cycle as another warp was selected. A high number of not selected warps typically means you have sufficient warps to cover warp latencies and you may consider reducing the number of active warps to possibly increase cache coherence and data locality. |
> | `smsp__warpidsamp_warps_issue_stalled_selected_not_issued` | Warp was selected by the micro scheduler and issued an instruction. |
> | `smsp__warpidsamp_warps_issue_stalled_short_scoreboard_not_issued` | Warp was stalled waiting for a scoreboard dependency on a MIO (memory input/output) operation (not to L1TEX). The primary reason for a high number of stalls due to short scoreboards is typically memory operations to shared memory. Other reasons include frequent execution of special math instructions (e.g. MUFU) or dynamic branching (e.g. BRX, JMX). Consult the Memory Workload Analysis section to verify if there are shared memory operations and reduce bank conflicts, if reported. Assigning frequently accessed values to variables can assist the compiler in using low-latency registers instead of direct memory accesses. |
> | `smsp__warpidsamp_warps_issue_stalled_sleeping_not_issued` | Warp was stalled due to all threads in the warp being in the blocked, yielded, or sleep state. Reduce the number of executed NANOSLEEP instructions, lower the specified time delay, and attempt to group threads in a way that multiple threads in a warp sleep at the same time. |
> | `smsp__warpidsamp_warps_issue_stalled_tex_throttle_not_issued` | Warp was stalled waiting for the L1 instruction queue for texture operations to be not full. This stall reason is high in cases of extreme utilization of the L1TEX pipeline. Try issuing fewer texture fetches, surface loads, surface stores, or decoupled math operations. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions. Consider converting texture lookups or surface loads into global memory lookups. Texture can accept four threads’ requests per cycle, whereas global accepts 32 threads. |
> | `smsp__warpidsamp_warps_issue_stalled_wait_not_issued` | Warp was stalled waiting on a fixed latency execution dependency. Typically, this stall reason should be very low and only shows up as a top contributor in already highly optimized kernels. Try to hide the corresponding instruction latencies by increasing the number of active warps, restructuring the code or unrolling loops. Furthermore, consider switching to lower-latency instructions, e.g. by making use of fast math compiler options. |
> | `smsp__warpidsamp_warps_issue_stalled_warpgroup_arrive_not_issued` | Warp was stalled waiting on a WARPGROUP.ARRIVES or WARPGROUP.WAIT instruction. |
>
> Warp Stalls per Warp ID (Not Issued) {#id52 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.11. Source Metrics

Most are collected using SASS-patching [4](#fmetricssass2). These metrics have instance values mapping from function address (uint64) to associated values (uint64). Metrics `memory_[access_]type` map to string values.

> |  |  |
> |----|----|
> | `branch_inst_executed` | Number of unique branch targets assigned to the instruction, including both divergent and uniform branches. |
> | `derived__avg_thread_executed` | Average number of thread-level executed instructions per warp (regardless of their predicate). Computed as: thread_inst_executed / inst_executed |
> | `derived__avg_thread_executed_true` | Average number of predicated-on thread-level executed instructions per warp. Computed as: thread_inst_executed_true / inst_executed |
> | `derived__derivative_avg_thread_executed_true` | Derivative of the derived\_\_avg_thread_executed_true metric. Difference between the current and the previous address in average number of predicated-on thread-level executed instructions per warp. |
> | `derived__local_spilling_requests` | Number of executed instructions and requests made to L1 for register spilling to local memory. |
> | `derived__local_spilling_requests_pct` | Percentage of total local memory requests to L1 that are due to register spilling. |
> | `derived__memory_l1_conflicts_shared_nway` | Average N-way conflict in L1 per shared memory instruction. A 1-way access has no conflicts and resolves in a single pass. Computed as: memory_l1_wavefronts_shared / inst_executed |
> | `derived__memory_l1_wavefronts_shared_excessive` | Excessive number of wavefronts in L1 from shared memory instructions, because not all not predicated-off threads performed the operation. |
> | `derived__memory_l2_theoretical_sectors_global_excessive` | Excessive theoretical number of sectors requested in L2 from global memory instructions, because not all not predicated-off threads performed the operation. |
> | `derived__shared_spilling_requests` | Number of executed instructions and requests made to shared memory for register spilling. |
> | `derived__shared_spilling_requests_pct` | Percentage of total requests to shared memory that are due to register spilling. |
> | `inst_executed` | Number of warp-level executed instructions, ignoring instruction predicates. Warp-level means the values increased by one per individual warp executing the instruction, independent of the number of participating threads within each warp. |
> | `memory_access_size_type` | The size of the memory access, in bits. |
> | `memory_access_type` | The type of memory access (e.g. load or store). |
> | `memory_l1_tag_requests_global` | Number of L1 tag requests generated by global memory instructions. |
> | `memory_l1_wavefronts_shared` | Number of wavefronts in L1 from shared memory instructions. |
> | `memory_l1_wavefronts_shared_ideal` | Ideal number of wavefronts in L1 from shared memory instructions, assuming each not predicated-off thread performed the operation. |
> | `memory_l2_theoretical_sectors_global` | Theoretical number of sectors requested in L2 from global memory instructions. |
> | `memory_l2_theoretical_sectors_global_ideal` | Ideal number of sectors requested in L2 from global memory instructions, assuming each not predicated-off thread performed the operation. |
> | `memory_l2_theoretical_sectors_local` | Theoretical number of sectors requested in L2 from local memory instructions. |
> | `memory_type` | The accessed address space (global/local/shared). |
> | `smsp__branch_targets_threads_divergent` | Number of divergent branch targets, including fallthrough. Incremented only when there are two or more active threads with divergent targets. |
> | `smsp__branch_targets_threads_uniform` | Number of uniform branch execution, including fallthrough, where all active threads selected the same branch target. |
> | `smsp__pcsamp_sample_count` | Number of collected warp state samples per program counter. This metric is collected using warp sampling. |
> | `thread_inst_executed` | Number of thread-level executed instructions, regardless of predicate presence or evaluation. |
> | `thread_inst_executed_true` | Number of thread-level executed instructions, where the instruction predicate evaluated to true, or no predicate was given. |
>
> Source Metrics {#id53 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.12. L2 Cache Eviction Metrics

> |  |  |
> |----|----|
> | `smsp__sass_inst_executed_memdesc_explicit_evict_type` | L2 cache eviction policy types. |
> | `smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_first` | Number of warp-level executed instructions with L2 cache eviction hit property ‘first’. |
> | `smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_last` | Number of warp-level executed instructions with L2 cache eviction hit property ‘last’. |
> | `smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_normal` | Number of warp-level executed instructions with L2 cache eviction hit property ‘normal’. |
> | `smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_normal_demote` | Number of warp-level executed instructions with L2 cache eviction hit property ‘normal demote’. |
> | `smsp__sass_inst_executed_memdesc_explicit_missprop_evict_first` | Number of warp-level executed instructions with L2 cache eviction miss property ‘first’. |
> | `smsp__sass_inst_executed_memdesc_explicit_missprop_evict_normal` | Number of warp-level executed instructions with L2 cache eviction miss property ‘normal’. |
>
> L2 Cache Eviction Metrics {#id54 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.13. Instructions Per Opcode Metrics

Collected using SASS-patching. These metrics have instance values mapping from the SASS opcode or opcode category (string) to the number of executions (uint64).

> |  |  |
> |----|----|
> | `sass__inst_executed_per_opcode` | Number of warp-level executed instructions, instanced by basic SASS opcode. |
> | `sass__inst_executed_per_opcode_category` | Number of warp-level executed instructions, instanced by SASS opcode category. |
> | `sass__inst_executed_per_opcode_pipeline` | Estimated number of warp-level executed instructions, instanced by SASS opcode pipeline. Some instructions can execute in one of multiple pipelines, and the dynamic assignment is not taken into account here. |
> | `sass__inst_executed_per_opcode_with_modifier_all` | Number of warp-level executed instructions, instanced by all SASS opcode modifiers. |
> | `sass__inst_executed_per_opcode_with_modifier_selective` | Number of warp-level executed instructions, instanced by selective SASS opcode modifiers. |
> | `sass__thread_inst_executed_per_opcode_category` | Number of thread-level executed instructions, instanced by SASS opcode category. |
> | `sass__thread_inst_executed_per_opcode_pipeline` | Estimated number of thread-level executed instructions, instanced by SASS opcode pipeline. Some instructions can execute in one of multiple pipelines, and the dynamic assignment is not taken into account here. |
> | `sass__thread_inst_executed_true_per_opcode` | Number of thread-level executed instructions, instanced by basic SASS opcode. |
> | `sass__thread_inst_executed_true_per_opcode_with_modifier_all` | Number of thread-level executed instructions, instanced by all SASS opcode modifiers. |
> | `sass__thread_inst_executed_true_per_opcode_with_modifier_selective` | Number of thread-level executed instructions, instanced by selective SASS opcode modifiers. |
>
> Instructions Per Opcode Metrics {#id55 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.14. SASS Unit-Level Instructions Executed Metrics

Number of unit-level warp instructions executed.

> |  |  |
> |----|----|
> | `sass__inst_executed_global_loads` | Number of global memory load instructions executed. |
> | `sass__inst_executed_global_stores` | Number of global memory store instructions executed. |
> | `sass__inst_executed_local_loads` | Number of local memory load instructions executed. |
> | `sass__inst_executed_local_stores` | Number of local memory store instructions executed. |
> | `sass__inst_executed_register_spilling` | Number of store and load instructions executed as a result of register spilling. |
> | `sass__inst_executed_register_spilling_mem_local` | Number of store and load instructions executed as a result of register spilling to local memory. |
> | `sass__inst_executed_register_spilling_mem_shared` | Number of store and load instructions executed as a result of register spilling to shared memory. |
> | `sass__inst_executed_register_spilling_op_read` | Number of register read instructions executed as a result of register spilling. |
> | `sass__inst_executed_register_spilling_op_write` | Number of register write instructions executed as a result of register spilling. |
> | `sass__inst_executed_shared_loads` | Number of shared memory load instructions executed. |
> | `sass__inst_executed_shared_stores` | Number of shared memory store instructions executed. |
> | `sass__size` | Total size in bytes of all SASS instructions. |
> | `sass__size_inst_executed` | Total size in bytes of executed SASS instructions. |
> | `sass__size_thread_inst_executed_true` | Total size in bytes of executed SASS thread instructions with predicate true. |
>
> SASS Unit-Level Instructions Executed Metrics {#id56 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.15. Metric Groups

> |  |  |
> |----|----|
> | `group:memory__chart` | Group of metrics for the workload analysis chart. |
> | `group:memory__dram_table` | Group of metrics for the device memory workload analysis table. |
> | `group:memory__first_level_cache_table` | Group of metrics for the L1/TEX cache workload analysis table. |
> | `group:memory__l2_cache_evict_policy_table` | Group of metrics for the L2 cache eviction policies table. |
> | `group:memory__l2_cache_table` | Group of metrics for the L2 cache workload analysis table. |
> | `group:memory__shared_table` | Group of metrics for the shared memory workload analysis table. |
> | `group:smsp__pcsamp_warp_stall_reasons` | Group of metrics for the number of samples from the warp sampler per program location. |
> | `group:smsp__pcsamp_warp_stall_reasons_not_issued` | Group of metrics for the number of samples from the warp sampler per program location on cycles the warp scheduler issued no instructions. |
> | `group:smsp__pmwarpsamp_warp_stall_reasons` | Group of metrics for the number of samples from the warp sampler per program location collected with PM sampling. These metrics can not yet be collected from the command line. |
> | `group:smsp__warpidsamp_warp_stall_reasons` | Group of metrics for the number of samples from the warp sampler per warp ID. |
> | `group:smsp__warpidsamp_warp_stall_reasons_not_issued` | Group of metrics for the number of samples from the warp sampler per warp ID on cycles the warp scheduler issued no instructions. |
>
> Metric Groups {#id57 .colwidths-auto .table-no-stripes .docutils .align-default}

### 2.4.16. Profiler Metrics

Metrics generated by the tool itself to inform about statistics or problems during profiling.

> [TABLE]
>
> Profiler Metrics {#id58 .colwidths-auto .table-no-stripes .docutils .align-default}

Footnotes

[4](#id6)  
Instruction-level source metrics do not require profiling permissions on the target device when collected through the command line interface.

## 2.5. Sampling

NVIDIA Nsight Compute can collect certain performance data via sampling at fixed intervals.

### 2.5.1. PM Sampling

NVIDIA Nsight Compute supports collecting many metrics by sampling the GPU’s performance monitors (PM) periodically at fixed intervals. The resulting metrics are [instanced](index.html#instanced-metrics), with each sample being composed of its value and the (GPU) timestamp when it was collected. This allows the tool to visualize the data on a [timeline](../NsightCompute/index.html#profiler-report-details-page) that helps you understand how the behavior of the profiled workload changes during its runtime.

Metrics collected with PM sampling have instance values mapping from their sample timestamp (in ns) to their sample value. When logically possible, the non-instanced value of the metric represents the aggregate across all instances. The aggregation operation (e.g. sum, average) depends on the metric structure.

A metric is collected using PM sampling in the following cases:

- The metric name has the `pmsampling:` [prefix](../NsightComputeCli/index.html#command-line-options-profile).

- The metric name includes a valid `Triage` group.

- The metric is requested in a section’s `Timeline` field. Prefixing the metric with `pmsampling:` is still recommended in this case to avoid conflicts with profiler metrics of the same name collected e.g. by other sections.



Example PM sampling timeline for an application using tensor cores. The timeline shows a typical tail effect starting at ~6.5ms where there is not sufficient work anymore to fill all SMs. It also allows you to see the relationship between compute (SM Tensor Pipe Throughput) and data loading (DRAM Throughput and L2 Hit Rate). Whenever the L2 cache doesn’t have the needed data anymore readily available, its hit rate decreases, DRAM Throughput increases as the data needs to be loaded from global memory, and the SM Tensor Pipe Throughput decreases as the SMs are waiting for the data to arrive.

#### Support

| Architecture      | Support       | Sampling Intervals           |
|-------------------|---------------|------------------------------|
| Volta and earlier | Not supported | n/a                          |
| TU10x-GA100       | Supported     | \>= 20000 cycles             |
| GA10x and later   | Supported     | \>= 1000 ns [5](#fsampling1) |

Supported Architectures for PM Sampling {#id60 .docutils .align-default}

PM sampling is supported on all platforms except vGPU. See below for further limitations that apply to [context switch trace](index.html#ctx-switch-trace). You can [query](../NsightComputeCli/index.html#command-line-options-profile) the list of metrics available to PM sampling using the `--query-metrics-collection`` ``pmsampling` option. Note though that while all listed metrics are available to the PM sampler, only those requiring a single pass can be collected.

#### Context Switch Trace

Since this data collection samples across the entire GPU device, the tool concurrently collects a *context switch trace*. The trace is stored as a separate, instanced [metric](index.html#profiler-metrics). It tracks when the context of interest was active and can be used to filter the sampling metric to only relevant instances and to better align metrics from multiple passes on the timeline. While it’s generally preferable to have this trace collected, it can be disabled using an [environment variable](../NsightComputeCli/index.html#environment-variables).

Note that context switch trace is not supported on Windows Subsystem for Linux (WSL), Multi-Instance GPU (MIG), Multi-Process Service (MPS) with multiple clients, or on mobile platforms. Context switch trace in containers is supported with CUDA 12.7 drivers or newer.

#### Counter Domains

PM sampling metrics are composed of one or more raw counter dependencies internally. If metrics in the same pass share such a dependency, it is only collected once. Each counter is associated with a *counter domain*, which describes how and where in the hardware the counter is collected. There is a limited number of counters in each domain that can be collected concurrently in the same pass, and the number may vary, depending on the selected counters.

Selecting counters from different domains has the possibility that more metric dependencies can be fit into the same pass. Furthermore, some counters can be collected through different domains, and the domain may be chosen by the tool or the user.

When [querying](../NsightComputeCli/index.html#command-line-options-profile) the PM sampling metric collection, the required and optional domains for a metric’s counter dependencies are shown. E.g., for `l1tex__throughput`` ``gpu_sm_a,[gpu_sm_b,gpu_sm_c]`, the domain `gpu_sm_a` is required and one of the optional domains `[gpu_sm_b,gpu_sm_c]` must be chosen for this metric to be collectable. Counter domains can only be selected explicitly in [section files](../CustomizationGuide/index.html#counter-domains), using one or more instances of the `CtrDomains:`` ``"<domain>"` field for PM sampling metrics.

Note that most users should be able to rely on the tool’s automatic selection of counter domains, or the pre-configured domains in section files.

#### Known Issues

You should be aware of the following potential issues when using PM sampling:

Data collected from different passes may not align perfectly. In the below example, each group of warp stall reasons was collected in a separate pass. The total number of warps doesn’t change during the workload execution, i.e., at one timestamp, data from all passes should sum up to the same total number of warps. However, due to small misalignments, the aggregate row shows several peaks that are likely flatter in the real execution.



When collecting PM sampling metrics on MPS, the *workload execution* row only shows workloads from the primary client.

Footnotes

[5](#id7)  
For some chips or configurations, the minimum sampling interval may be higher.

### 2.5.2. Warp Sampling

NVIDIA Nsight Compute supports periodic sampling of the warp program counter and warp scheduler state. At a fixed interval of cycles, the sampler in each streaming multiprocessor selects an active warp and outputs the program counter and the warp scheduler state. The tool selects the minimum interval for the device. On small devices, this can be every 32 cycles. On larger chips with more multiprocessors, this may be 2048 cycles. The sampler selects a random active warp. On the same cycle the scheduler may select a different warp to issue.

The resulting metrics are correlated with the individual executed instructions but don’t have any time resolution.

See the *Warp Stall Reasons* tables in the [Metrics Reference](index.html#metrics-reference) for a description of the individual warp scheduler states.

## 2.6. Reproducibility

In order to provide actionable and deterministic results across application runs, NVIDIA Nsight Compute applies various methods to adjust how metrics are collected. This includes [serializing](index.html#serialization) kernel launches, [purging GPU caches](index.html#cache-control) before each kernel replay or [adjusting GPU clocks](index.html#clock-control).

### 2.6.1. Serialization

NVIDIA Nsight Compute serializes kernel launches within the profiled application, potentially across multiple processes profiled by one or more instances of the tool at the same time.

Serialization across processes is necessary since for the collection of HW performance metrics, some GPU and driver objects can only be acquired by a single process at a time. This is done on a per-CUDA device or MIG instance [6](#serialization-mig-f1) basis, meaning only one process can profile a given device at a time. To achieve this, on Linux, lock files `TMPDIR/nvidia/nsight_compute/lock.<UUID>` are used where `UUID` is the unique identifier of a given CUDA device or MIG instance. On other platforms, lock files are located at `TMPDIR/nsight-compute-lock.<UUID>`. On Windows, `TMPDIR` is the path returned by the Windows `GetTempPath` API function. On QNX, it is `/var/nvidia`, and `/tmp` on other platforms. The `TMPDIR` path can be overridden by setting the environment variables `TMPDIR`, `TMP`, `TEMP`, or `TEMPDIR`, read in that order.

Serialization within the process is required for most metrics to be mapped to the proper kernel. In addition, without serialization, performance metric values might vary widely if kernel execute concurrently on the same device.

It is possible to force a single global lock to be used for all processes - disabling concurrent profiling across devices - by setting the environment variable `NV_COMPUTE_PROFILER_DISABLE_CONCURRENT_PROFILING`. Refer to the [Environment Variables](index.html#environment-variables) entry on possible workarounds.

[6](#id8)  
Concurrent MIG profiling is not supported on GA100. Only one process will be allowed to profile on this chip when MIG is enabled.

### 2.6.2. Clock Control

For many metrics, their value is directly influenced by the current GPU SM and memory clock frequencies. For example, if a kernel instance is profiled that has prior kernel executions in the application, the GPU might already be in a higher clocked state and the measured kernel duration, along with other metrics, will be affected. Likewise, if a kernel instance is the first kernel to be launched in the application, GPU clocks will regularly be lower. In addition, due to kernel replay, the metric value might depend on which replay pass it is collected in, as later passes would result in higher clock states.

To mitigate this non-determinism, NVIDIA Nsight Compute attempts to limit GPU clock frequencies to their *base* value. As a result, metric values are less impacted by the location of the kernel in the application, or by the number of the specific replay pass.

However, this behavior might be undesirable for analysis of the kernel, e.g. in cases where an external tool is used to fix clock frequencies, or where the behavior of the kernel within the application is analyzed. To solve this, users can adjust the `--clock-control` option to specify if any clock frequencies should be fixed by the tool.

Note

Thermal throttling directed by the hardware or driver cannot be controlled by the tool and always overrides any selected options.

Factors affecting Clock Control:

- On mobile targets, e.g. L4T or QNX, there may be variations in profiling results due the inability for the tool to lock clocks. Using Nsight Compute’s `--clock-control` to set the GPU clocks will fail or will be silently ignored when profiling on a GPU partition.

- On L4T, you can use the jetson_clocks script to lock the clocks at their maximums during profiling.

- On Linux (aarch64 sbsa), clock control is not supported with Nsight Compute for the following GPUs: GA10b (Orin), GB10b (Thor). Instead, for these GPUs, you can manually lock and unlock the clocks with the following commands:

  - Save the existing min/max frequency limitation into /tmp/devfreq.config file

    ``` bash
    grep "" /sys/class/devfreq/gpu*/{min,max}_freq | tee /tmp/devfreq.config
    ```

  - Lock the frequency to the specified rate in HZ (e.g. 1575000000)

    ``` bash
    tee /sys/class/devfreq/gpu*/{min,max}_freq <<< 1575000000
    ```

  - Unlock the frequency to restore back the min/max frequency limitation

    ``` bash
    awk -F: '{print $2 > $1}' /tmp/devfreq.config
    ```

- See the [Special Configurations](index.html#special-configurations) section for MIG and vGPU clock control.

### 2.6.3. Cache Control

As explained in [Kernel Replay](index.html#kernel-replay), the kernel might need to be replayed multiple times to collect all requested metrics. While NVIDIA Nsight Compute can save and restore the contents of GPU device memory accessed by the kernel for each pass, it cannot do the same for the contents of HW caches, such as e.g. the L1 and L2 cache.

This can have the effect that later replay passes might have better or worse performance than e.g. the first pass, as the caches could already be primed with the data last accessed by the kernel. Similarly, the values of HW performance counters collected by the first pass might depend on which kernels, if any, were executed prior to the measured kernel launch.

In order to make HW performance counter value more deterministic, NVIDIA Nsight Compute by default flushes all GPU caches before each replay pass. As a result, in each pass, the kernel will access a clean cache and the behavior will be as if the kernel was executed in complete isolation.

This behavior might be undesirable for performance analysis, especially if the measurement focuses on a kernel within a larger application execution, and if the collected data targets cache-centric metrics. In this case, you can use `--cache-control`` ``none` to disable flushing of any HW cache by the tool.

### 2.6.4. Persistence Mode

The NVIDIA kernel mode driver must be running and connected to a target GPU device before any user interactions with that device can take place. The driver behavior differs depending on the OS. Generally, on Linux, if the kernel mode driver is not already running or connected to a target GPU, the invocation of any program that attempts to interact with that GPU will transparently cause the driver to load and/or initialize the GPU. When all GPU clients terminate the driver will then deinitialize the GPU.

If [persistence mode](https://docs.nvidia.com/deploy/driver-persistence/index.html) is not enabled (as part of the OS, or by the user), applications triggering GPU initialization may incur a short startup cost. In addition, on some configurations, there may also be a shutdown cost when the GPU is de-initialized at the end of the application.

It is recommended to enable persistence mode on applicable operating systems before profiling with NVIDIA Nsight Compute for more consistent application behavior.

## 2.7. Special Configurations

### 2.7.1. Multi Instance GPU

Multi-Instance GPU (MIG) is a feature that allows a GPU to be partitioned into multiple CUDA devices. The partitioning is carried out on two levels: First, a GPU can be split into one or multiple GPU Instances. Each GPU Instance claims ownership of one or more streaming multiprocessors (SM), a subset of the overall GPU memory, and possibly other GPU resources, such as the video encoders/decoders. Second, each GPU Instance can be further partitioned into one or more Compute Instances. Each Compute Instance has exclusive ownership of its assigned SMs of the GPU Instance. However, all Compute Instances within a GPU Instance share the GPU Instance’s memory and memory bandwidth. Every Compute Instance acts and operates as a CUDA device with a unique device ID. See the driver release notes as well as the documentation for the `nvidia-smi` CLI tool for more information on how to configure MIG instances.

For profiling, a Compute Instance can be of one of two types: *isolated* or *shared*.

An *isolated* Compute Instance owns all of its assigned resources and does not share any GPU unit with another Compute Instance. In other words, the Compute Instance is the same size as its parent GPU Instance and consequently does not have any other sibling Compute Instances. Profiling works as usual for isolated Compute Instances.

A *shared* Compute Instance uses GPU resources that can potentially also be accessed by other Compute Instances in the same GPU Instance. Due to this resource sharing, collecting profiling data from those shared units is not permitted. Attempts to collect metrics from a shared unit fail with an error message of `==ERROR==`` ``Failed`` ``to`` ``access`` ``the`` ``following`` ``metrics.`` ``When`` ``profiling`` ``on`` ``a`` ``MIG`` ``instance,`` ``it`` ``is`` ``not`` ``possible`` ``to`` ``collect`` ``metrics`` ``from`` ``GPU`` ``units`` ``that`` ``are`` ``shared`` ``with`` ``other`` ``MIG`` ``instances` followed by the list of failing metrics. Collecting only metrics from GPU units that are exclusively owned by a shared Compute Instance is still possible.

#### Locking Clocks

NVIDIA Nsight Compute is not able to set the clock frequency on any Compute Instance for profiling. You can continue analyzing kernels without fixed clock frequencies (using `--clock-control`` ``none`; see [here](index.html#clock-control) for more details). If you have sufficient permissions, `nvidia-smi` can be used to configure a fixed frequency for the whole GPU by calling `nvidia-smi`` ``--lock-gpu-clocks=tdp,tdp`. This sets the GPU clocks to the base TDP frequency until you reset the clocks by calling `nvidia-smi`` ``--reset-gpu-clocks`.

#### MIG on Baremetal (non-vGPU)

All Compute Instances on a GPU share the same clock frequencies.

#### MIG on NVIDIA vGPU

Enabling profiling for a VM gives the VM access to the GPU’s global performance counters, which may include activity from other VMs executing on the same GPU. Enabling profiling for a VM also allows the VM to lock clocks on the GPU, which impacts all other VMs executing on the same GPU, including MIG Compute Instances.

### 2.7.2. CUDA Green Contexts

[CUDA Green Contexts](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS) is a feature of the CUDA driver API that allows spacial partitioning of a GPU by assigning a set of resources to a CUDA context on which the context should operate on. NVIDIA Nsight Compute supports profiling applications using CUDA Green Contexts by enabling collection of the same metrics as described in the [Metrics Reference](metrics-reference.html). In particular, any combination of *Workload Type* and *Replay Mode* within the [compatibility matrix](index.html#compatibility) is also supported for Green Context applications. For a list of API functions supported in range-based replay modes refer to the *Green Contexts* section in the [supported APIs](index.html#supported-apis).

Unlike with regular CUDA contexts, when using Green Contexts, metrics are subdivided into two categories: those that can be attributed to a Green Context directly, and those that cannot. The former comprise of metrics that are collected on hardware units that are directly assigned to the Green Context, such as SMs or the L1 cache, where the latter are composed of metrics that do not give exclusive access to a Green Context, such as the L2 cache. The former will be referred to as Green-Context *attributable* metrics and the latter as *non-attributable* metrics.

Since it is generally more insightful to analyze the performance of a Green Context application relative to the resources it was assigned, instead of the entire GPU, *attributable* metrics are scaled to the number of SMs of the Green Context. [7](#green-context-f1) This is not the case for *non-attributable* metrics. Refer to the *supported driver versions* below for more information on the minimum driver version required for this feature.

To get an overview of how Green Contexts are supported in the UI, including ways to distinguish *attributable* and *non-attributable* metrics, refer to the [Green Contexts support](../NsightCompute/index.html#green-contexts-support) section. On the CLI, –print-metric-attribution may be used to display the attribution levels, see the [Command Line Options](../NsightComputeCli/index.html#console-output).

#### Supported driver versions

Starting with NVIDIA Nsight Compute 2024.1, profiling of applications using CUDA Green Contexts is fully supported. This requires a minimum driver version of 550. Note, however, that with this version metrics were always scaled to the number of all SMs available on the GPU.

From NVIDIA Nsight Compute 2024.3 onwards, *attributable* metrics are scaled to the number of SMs used by the Green Context. This requires a minimum driver version of 560.

Footnotes

[7](#id9)  
Although not intended as the primary use-case, different Green Contexts can be initialized with overlapping resources. When executing and profiling such Green Contexts concurrently, *attributable* metrics may contain contributions from multiple Green Contexts, although the same metric scaling behavior as described in this section would still apply.

### 2.7.3. Multi-Process Service

The Multi-Process Service (MPS) runtime architecture enables cooperative multi-process CUDA applications to utilize Hyper-Q capabilities on NVIDIA GPUs. This allows CUDA kernels to be processed concurrently on the same GPU, benefitting performance when the GPU compute capacity is underutilized by a single application process.

Nsight Compute can be used to profile how the GPU is utilized while executing the work from all MPS clients concurrently. It does generally not support isolating the performance of individual clients.

#### Launching MPS Applications

When profiling MPS applications, Nsight Compute differentiates between MPS client processes and the ncu control process. Client (application) processes must be launched using the `ncu`` ``--mps`` ``client` command. Each client will be suspended, waiting for the control process to attach. The ncu control process is launched as `ncu`` ``--mps`` ``control`. It will wait until the expected number of client processes is active, attach to them and start profiling. The arguments to the control process specify how to select and replay workloads (kernel or range), which metrics to collect, the output report file, etc. The ncu processes launching the MPS client processes act only as simple launchers. However, if the client processes are expected to trace NVTX (e.g. for defining ranges by NVTX markers), the `--nvtx` flag is needed. A typical set of launch commands may hence look like this:

``` text
$ ncu --mps client --nvtx ./my_mps_client 1
$ ncu --mps client --nvtx ./my_mps_client 2
$ ncu --mps control --mps-num-clients 2 --replay-mode range --nvtx-include INNER_RANGE --set full -f -o mps-report
```

If only a single MPS client process needs to be launched, it can be started directly from the Nsight Compute control process:

``` text
$ ncu --mps control --replay-mode range --nvtx-include INNER_RANGE ./my_mps_client
```

While Nsight Compute supports profiling MPS applications with `--replay-mode`` ``kernel`, it is recommended to use `--replay-mode`` ``range` whenever possible. In the former mode, each MPS client can only contribute a single kernel launch.

Profiling MPS applications is only supported in the command line interface (CLI). Collected reports can be analyzed in the CLI and UI.

#### Observation Window

[](../_images/mps-observation-window-range.png)

When no primary client is specified explicitly using `primary-client`, the observation window is the full window across all participating clients, regardless which one is chosen as the primary at runtime. This applies to both `kernel` and `range` replay modes. Note that all ranges are start-aligned to allow for deterministic replays.

[](../_images/mps-observation-window-kernel.png)

When a primary client is specified explicitly using `primary-client`, the observation window is the limited to the duration of this client, as long as it participates in the profile. All concurrent clients are still included for the duration of the window. This applies to both `kernel` and `range` replay modes. Note that all ranges are start-aligned to allow for deterministic replays.

#### Data Collection

Data collection for the individual metric groups is as follows:

- **Hardware Counters / SMSP**: Supported

- **Unit-Level Source**: Supported

- **Instruction-Level Source**: Supported, but source and metric data is only shown for the primary client.

- **Launch**: Supported

- **Warp Sampling**: Supported, but only attributed to the primary client’s source.

- **PM Sampling**: Supported

#### Limitations

- Only replay modes `kernel` and `range` are supported.

- Replay mode `kernel` only supports a narrow observation window for the primary client.

- All clients are expected to behave homogeneous in the sense that they use the same NVTX or start/stop markers around matching ranges.

- The list of workload filter options during collection is reduced. Only the following are supported: `kernel-name`, `kernel-name-base`, `launch-count`, `launch-skip`, `nvtx-push-pop-scope`.

## 2.8. Metric Distributor

The Metric Distributor distributes the workload of metric collection across multiple NCU processes, reducing the total number of passes required by the profiler. The Metric Distributor is particularly useful when a large number of metrics need to be collected and multiple GPUs are available in the system. The actual reduction in profiling overhead depends on the total number of metrics, the number of available GPUs, and the types of metrics being collected.

To enable this feature, specify the number of distribution groups with `--metric-distribution-groups`` ``<N>`. The Metric Distributor will divide the set of metrics to be collected into N groups and will assign each group to a different NCU process.

- The Metric Distributor parallelizes the metric collection task across multiple GPUs. As a result, each NCU process generates a partial report file (e.g., `report0.ncu-rep`). These partial reports **must be merged** into a single, complete report using [Report Merge Tool](../NsightCompute/index.html#report-merge-tool).

- For the distributor to work correctly, all participating GPUs must have an identical chip architecture.

- The Metric Distributor requires the communicator option to be set to either `none` or `tcp`. See [Multi-Process Support](../NsightComputeCli/index.html#multi-process).

**Usage and Examples**

- To profile a single-GPU application on a shared-memory machine with multiple GPUs, set `--communicator`` ``none`. NCU will automatically create processes for the available devices.

``` bash
# Distribute metric collection across 4 groups on a shared-memory machine.
# This generates partial reports: report0.ncu-rep through report3.ncu-rep.
ncu --set full --metric-distribution-groups 4 --communicator none -o report <app> [app arguments]

# After profiling, merge the partial reports from a report directory into a final report.
$NCU_INSTALL_PATH/extras/ReportUtils/ReportMergeTool -i <report directory> -o final_report
```

- To profile single or multi-GPU applications with the TCP communicator, set `--communicator`` ``tcp`. With this option, peer NCU processes should be manually launched if the application is not already a multi-process application (e.g., MPI, NCCL). One of these processes acts as the server, and the others connect as clients. In addition, the total number of peers with `--communicator-tcp-num-peers` should be specified. This configuration allows for data redundancy. By using more GPUs than distribution groups (i.e., `--communicator-tcp-num-peers` \> `--metric-distribution-groups`), the same metric groups can be collected redundantly by multiple GPUs, increasing the robustness of the profiling data.

``` bash
# Example: Use 4 GPUs to collect 2 metric groups for redundancy.
# Each metric group will be collected by 2 GPUs.
ncu --set full --metric-distribution-groups 2 --communicator tcp --communicator-tcp-num-peers 4 <app> [app arguments]

# Example: Use with a 4-rank MPI application.
# Each rank saves a unique report using its MPI rank in the filename.
mpirun -np 4 ncu --set full --metric-distribution-groups 4 --communicator tcp --communicator-tcp-num-peers 4 \
        -o report_%q{OMPI_COMM_WORLD_RANK} <app> [app arguments]
```

## 2.9. Roofline Charts

Roofline charts provide a very helpful way to visualize achieved performance on complex processing units, like GPUs. This section introduces the Roofline charts that are presented within a profile report.

Note

The values of operations/second calculated from this NVIDIA developer tools site and generated by using the data center monitoring tools are not calculated in the same way as the operations/second used for export control purposes and should not be relied upon to assess performance against the export control limits.

### 2.9.1. Overview

Kernel performance is not only dependent on the operational speed of the GPU. Since a kernel requires data to work on, performance is also dependent on the rate at which the GPU can feed data to the kernel. A typical roofline chart combines the peak performance and memory bandwidth of the GPU, with a metric called *Arithmetic Intensity* (a ratio between *Work* and *Memory Traffic*), into a single chart, to more realistically represent the achieved performance of the profiled kernel. A simple roofline chart might look like the following:



Roofline overview.

This chart actually shows two different rooflines. However, the following components can be identified for each:

- **Vertical Axis** - The vertical axis represents *Floating Point Operations per Second* (FLOPS). For GPUs this number can get quite large and so the numbers on this axis can be scaled for easier reading (as shown here). In order to better accommodate the range, this axis is rendered using a logarithmic scale.

- **Horizontal Axis** - The horizontal axis represents *Arithmetic Intensity*, which is the ratio between *Work* (expressed in floating point operations per second), and *Memory Traffic* (expressed in bytes per second). The resulting unit is in floating point operations per byte. This axis is also shown using a logarithmic scale.

- **Memory Bandwidth Boundary** - The memory bandwidth boundary is the *sloped* part of the roofline. By default, this slope is determined entirely by the memory transfer rate of the GPU but can be customized inside the *SpeedOfLight_RooflineChart.section* file if desired.

- **Peak Performance Boundary** - The peak performance boundary is the *flat* part of the roofline By default, this value is determined entirely by the peak performance of the GPU but can be customized inside the *SpeedOfLight_RooflineChart.section* file if desired.

- **Ridge Point** - The ridge point is the point at which the memory bandwidth boundary meets the peak performance boundary. This point is a useful reference when analyzing kernel performance.

- **Achieved Value** - The achieved value represents the performance of the profiled kernel. If baselines are being used, the roofline chart will also contain an achieved value for each baseline. The outline color of the plotted achieved value point can be used to determine from which baseline the point came.

### 2.9.2. Analysis

The roofline chart can be very helpful in guiding performance optimization efforts for a particular kernel.



Roofline anaysis.

As shown here, the *ridge point* partitions the roofline chart into two regions. The area shaded in blue under the sloped *Memory Bandwidth Boundary* is the *Memory Bound* region, while the area shaded in green under the *Peak Performance Boundary* is the *Compute Bound* region. The region in which the *achieved value* falls, determines the current limiting factor of kernel performance.

The distance from the *achieved value* to the respective roofline boundary (shown in this figure as a dotted white line), represents the opportunity for performance improvement. The closer the *achieved value* is to the roofline boundary, the more optimal is its performance. An *achieved value* that lies on the *Memory Bandwidth Boundary* but is not yet at the height of the *ridge point* would indicate that any further improvements in overall FLOP/s are only possible if the *Arithmetic Intensity* is increased at the same time.

Using the baseline feature in combination with roofline charts, is a good way to track optimization progress over a number of kernel executions.

## 2.10. Memory Chart

The *Memory Chart* shows a graphical, logical representation of performance data for memory subunits on and off the GPU. Performance data includes transfer sizes, hit rates, number of instructions or requests, etc.

### 2.10.1. Overview



Memory chart for an NVIDIA A100 GPU

#### Logical Units (green)

Logical units are shown in green (active) or grey (inactive).

- Kernel: The CUDA kernel executing on the GPU’s Streaming Multiprocessors

- Global: CUDA [global memory](index.html#metrics-hw-memory)

- Local: CUDA [local memory](index.html#metrics-hw-memory)

- Texture: CUDA [texture memory](index.html#metrics-hw-tex-surf)

- Surface: CUDA [surface memory](index.html#metrics-hw-tex-surf)

- Shared: CUDA [shared memory](index.html#metrics-hw-memory)

- Load Global Store Shared: Instructions loading directly from global into shared memory without intermediate register file access

#### Physical Units (blue)

Physical units are shown in blue (active) or grey (inactive).

- L1/TEX Cache: The [L1/Texture cache](index.html#metrics-hw-caches). The underlying physical memory is split between this cache and the user-managed *Shared Memory*.

- Shared Memory: CUDA’s user-managed [shared memory](index.html#metrics-hw-memory). The underlying physical memory is split between this and the *L1/TEX Cache*.

- L2 Cache: The [L2 cache](index.html#metrics-hw-caches)

- L2 Compression: The memory compression unit of the *L2 Cache*

- System Memory: Off-chip [system (CPU) memory](index.html#metrics-hw-memory)

- Device Memory: On-chip [device (GPU) memory](index.html#metrics-hw-memory) of the CUDA device that executes the kernel

- Peer Memory: On-chip [device (GPU) memory](index.html#metrics-hw-memory) of other CUDA devices

Depending on the exact GPU architecture, the exact set of shown units can vary, as not all GPUs have all units.

#### Links

Links between *Kernel* and other logical units represent the number of executed instructions (*Inst*) targeting the respective unit. For example, the link between *Kernel* and *Global* represents the instructions loading from or storing to the global memory space. Instructions using the NVIDIA A100’s *Load Global Store Shared* paradigm are shown separately, as their register or cache access behavior can be different from regular global loads or shared stores.

Links between logical units and blue, physical units represent the number of requests (*Req*) issued as a result of their respective instructions. For example, the link going from *L1/TEX Cache* to *Global* shows the number of requests generated due to global load instructions.

The color of each link represents the percentage of peak utilization of the corresponding communication path. The color legend to the right of the chart shows the applied color gradient from unused (0%) to operating at peak performance (100%). If a link is inactive, it is shown in grey color. Triangle markers to the left of the legend correspond to the links in the chart. The markers offer a more accurate value estimate for the achieved peak performances than the color gradient alone.

#### Ports

A unit often shares a common data port for incoming and outgoing traffic. While the links sharing a port might operate well below their individual peak performances, the unit’s data port may have already reached its peak. Port utilization is shown in the chart by colored rectangles inside the units located at the incoming and outgoing links. Ports use the same color gradient as the data links and have also a corresponding marker to the left of the legend. Inactive ports are shown in grey color.

#### Metrics

Metrics from this chart can be collected on the command line using `--set`` ``full`, `--section`` ``MemoryWorkloadAnalysis_Chart` or `--metrics`` ``group:memory__chart`. An example of the correlation between the peak values reported in the memory tables and the ports in the memory chart is shown below.



Mapping of peak values between memory tables and memory chart

## 2.11. Memory Tables

The *Memory Tables* show detailed metrics for the various memory HW units, such as shared memory, the caches, and device memory. For most table entries, you can hover over it to see the underlying metric name and description. Some entries are generated as derivatives from other cells, and do not show a metric name on their own, but the respective calculation. If a certain metric does not contribute to the generic derivative calculation, it is shown as *UNUSED* in the tooltip. You can hover over row or column headers to see a description of this part of the table.

### 2.11.1. Shared Memory



Example Shared Memory table, collected on an RTX 2080 Ti

#### Columns

[TABLE]

#### Rows

|  |  |
|----|----|
| `(Access`` ``Types)` | Shared memory access operations. |
| `Total` | The aggregate for all access types in the same column. |

#### Metrics

Metrics from this table can be collected on the command line using `--set`` ``full`, `--section`` ``MemoryWorkloadAnalysis_Tables` or `--metrics`` ``group:memory__shared_table`.

### 2.11.2. L1/TEX Cache



Example L1/TEX Cache memory table, collected on an RTX 2080 Ti



Model of the Global Load Pipeline for the L1TEX cache on GA100, mapped to the memory table.

#### Columns

[TABLE]

#### Rows

|  |  |
|----|----|
| `(Access`` ``Types)` | The various access types, e.g. loads from global memory or reduction operations on surface memory. |
| `Loads` | The aggregate of all load access types in the same column. |
| `Stores` | The aggregate of all store access types in the same column. |
| `Total` | The aggregate of all load and store access types in the same column. |

#### Metrics

Metrics from this table can be collected on the command line using `--set`` ``full`, `--section`` ``MemoryWorkloadAnalysis_Tables` or `--metrics`` ``group:memory__first_level_cache_table`.

### 2.11.3. L2 Cache



Example L2 Cache memory table, collected on an RTX 2080 Ti



Model of the L2 cache on GA100, mapped to the memory table.

#### Columns

|  |  |
|----|----|
| `Requests` | For each access type, the total number of [requests](index.html#metrics-decoder__metrics-quantities) made to the L2 cache. This correlates with the [Sector Misses to L2](index.html#memory-tables-l1__memory-tables-l1-columns) for the L1 cache. Each request accesses up to four sectors from a single 128 byte cache line. |
| `Sectors` | For each access type, the total number of [sectors](index.html#metrics-decoder__metrics-quantities) requested from the L2 cache. Each request accesses between one and four sectors. |
| `Sectors/Req` | The average ratio of sectors to requests for the L2 cache. For the same number of active threads in a warp, smaller numbers imply a more efficient memory access pattern. Smaller ratios indicate some degree of uniformity or overlapped loads within a cache line. Higher numbers can imply [uncoalesced memory accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) and will result in increased memory traffic. |
| `%`` ``Peak` | Percentage of peak sustained number of sectors. The “work package” in the L2 cache is a sector. Higher values imply a higher utilization of the unit and can show potential bottlenecks, as it does not necessarily indicate efficient usage. |
| `Hit`` ``Rate` | Hit rate (percentage of requested sectors that do not miss) in the L2 cache. Sectors that miss need to be requested from a later stage, thereby contributing to one of *Sector Misses to Device*, *Sector Misses to System*, or *Sector Misses to Peer*. Higher hit rates imply better performance due to lower access latencies, as the request can be served by L2 instead of a later stage. |
| `Bytes` | Total number of bytes requested from L2. This is identical to the number of sectors multiplied by 32 byte, since the minimum access size in L2 is one sector. |
| `Throughput` | Achieved L2 cache throughput in bytes per second. High values indicate high utilization of the unit. |
| `Sector`` ``Misses`` ``to`` ``Device` | Total number of sectors that miss in L2 and generate [subsequent requests](index.html#memory-tables-dram) in [device memory](index.html#metrics-hw-model__metrics-hw-memory). |
| `Sector`` ``Misses`` ``to`` ``System` | Total number of sectors that miss in L2 and generate subsequent requests in [system memory](index.html#metrics-hw-model__metrics-hw-memory). |
| `Sector`` ``Misses`` ``to`` ``Peer` | Total number of sectors that miss in L2 and generate subsequent requests in [peer memory](index.html#metrics-hw-model__metrics-hw-memory). |

#### Rows

|  |  |
|----|----|
| `(Access`` ``Types)` | The various access types, e.g. loads or reductions originating from L1 cache. |
| `L1/TEX`` ``Total` | Total for all operations originating from the L1 cache. |
| `ECC`` ``Total` | Total for all operations caused by ECC (Error Correction Code). If ECC is enabled, L2 write requests that partially modify a sector cause a corresponding sector load from DRAM. These additional load operations increase the sector misses of L2. |
| `L2`` ``Fabric`` ``Total` | Total for all operations across the L2 fabric connecting the two L2 partitions. This row is only shown for kernel launches on CUDA devices with L2 fabric. |
| `GPU`` ``Total` | Total for all operations across all clients of the L2 cache. Independent of having them split out separately in this table. |

#### Metrics

Metrics from this table can be collected on the command line using `--set`` ``full`, `--section`` ``MemoryWorkloadAnalysis_Tables` or `--metrics`` ``group:memory__l2_cache_table`.

### 2.11.4. L2 Cache Eviction Policies



Example L2 Cache Eviction Policies memory table, collected on an A100 GPU

#### Columns

|  |  |
|----|----|
| `First` | Number of sectors accessed in the L2 cache using the `evict_first` policy. Data cached with this policy will be first in the eviction priority order and will likely be evicted when cache eviction is required. This policy is suitable for streaming data. |
| `Hit`` ``Rate` | Cache hit rate for sector accesses in the L2 cache using the `evict_first` policy. |
| `Last` | Number of sectors accessed in the L2 cache using the `evict_last` policy. Data cached with this policy will be last in the eviction priority order and will likely be evicted only after other data with `evict_normal` or `evict_first` eviction policy is already evicted. This policy is suitable for data that should remain persistent in cache. |
| `Hit`` ``Rate` | Cache hit rate for sector accesses in the L2 cache using the `evict_last` policy. |
| `Normal` | Number of sectors accessed in the L2 cache using the `evict_normal` policy. This is the default policy. |
| `Hit`` ``Rate` | Cache hit rate for sector accesses in the L2 cache using the `evict_normal` policy. |
| `Normal`` ``Demote` | Number of sectors accessed in the L2 cache using the `evict_normal_demote` policy. |
| `Hit`` ``Rate` | Cache hit rate for sector accesses in the L2 cache using the `evict_normal_demote` policy. |

#### Rows

|  |  |
|----|----|
| `(Access`` ``Types)` | The various access types, e.g. loads or reductions, originating from L1 cache. |
| `L1/TEX`` ``Total` | Total for all operations originating from the L1 cache. |
| `L2`` ``Fabric`` ``Total` | Total for all operations across the L2 fabric connecting the two L2 partitions. This row is only shown for kernel launches on CUDA devices with L2 fabric. |
| `GPU`` ``Total` | Total for all operations across all clients of the L2 cache. Independent of having them split out separately in this table. |

#### Metrics

Metrics from this table can be collected on the command line using `--set`` ``full`, `--section`` ``MemoryWorkloadAnalysis_Tables` or `--metrics`` ``group:memory__l2_cache_evict_policy_table`. Note that this table is only available on GPUs with GA100 or newer.

### 2.11.5. Device Memory



Example Device Memory table, collected on an RTX 2080 Ti

#### Columns

|  |  |
|----|----|
| `Sectors` | For each access type, the total number of [sectors](index.html#metrics-decoder__metrics-quantities) requested from device memory. |
| `%`` ``Peak` | Percentage of peak device memory utilization. Higher values imply a higher utilization of the unit and can show potential bottlenecks, as it does not necessarily indicate efficient usage. |
| `Bytes` | Total number of bytes transferred between [L2 Cache](index.html#memory-tables-l2) and device memory. |
| `Throughput` | Achieved device memory throughput in bytes per second. High values indicate high utilization of the unit. |

#### Rows

|  |  |
|----|----|
| `(Access`` ``Types)` | Device memory loads and stores. |
| `Total` | The aggregate for all access types in the same column. |

#### Metrics

Metrics from this table can be collected on the command line using `--set`` ``full`, `--section`` ``MemoryWorkloadAnalysis_Tables` or `--metrics`` ``group:memory__dram_table`.

## 2.12. FAQ

- **n/a metric values**

  n/a means that the metric value is “not available”. The most common reason is that the requested metric does not exist. This can either be the result of a typo, or a missing [suffix](index.html#metrics-structure__metrics-examples). Verify the metric name against the output of of the `--query-metrics`NVIDIA Nsight Compute CLI option.

  If the metric name was copied (e.g. from an old version of this documentation), make sure that it does not contain zero-width unicode characters.

  Finally, the metric might simply not exist for the targeted GPU architecture. For example, the IMMA pipeline metric `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` is not available on GV100 chips.

- **Metric values outside the expected logical range**

  This includes e.g. percentages exceeding 100% or metrics reporting negative values. For further details, see [Range and Precision](index.html#range-and-precision).

- **ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device.**

  By default, NVIDIA drivers require elevated permissions to access GPU performance counters. On mobile platforms, profile as root/using sudo. On other platforms, you can either start profiling as root/using sudo, or by enabling non-admin profiling. For further details, see [https://developer.nvidia.com/ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

  On Windows Subsystem for Linux (WSL), access to NVIDIA GPU Performance Counters must be enabled in the NVIDIA Control Panel of the Windows host.

- **Unsupported GPU**

  This indicates that the GPU, on which the current kernel is launched, is not supported. See the [Release Notes](../ReleaseNotes/index.html#gpu-support) for a list of devices supported by your version of NVIDIA Nsight Compute. It can also indicate that the current *GPU configuration* is not supported. For example, NVIDIA Nsight Compute might not be able to profile GPUs in SLI configuration.

- **Connection error detected communicating with target application.**

  The inter-process connection to the profiled application unexpectedly dropped. This happens if the application is killed or signals an exception (e.g. segmentation fault).

- **Failed to connect. The target process may have exited.**

  This occurs if

  - the application does not call any CUDA API calls before it exits.

  - the application terminates early because it was started from the wrong working directory, or with the wrong arguments. In this case, check the details in the *Start Activity Dialog*.

  - the application crashes before calling any CUDA API calls.

  - the application launches child processes which use the CUDA. In this case, launch with the `--target-processes`` ``all` option.

- **The profiler returned an error code: (number)**

  For the non-interactive *Profile* activity, the NVIDIA Nsight Compute CLI is started to generate the report. If either the application exited with a non-zero return code, or the NVIDIA Nsight Compute CLI encountered an error itself, the resulting return code will be shown in this message.

  For example, if the application hit a segmentation fault (SIGSEGV) on Linux, it will likely return error code 11. All non-zero return codes are considered errors, so the message is also shown if the application exits with return code 1 during regular execution.

  To debug this issue, it can help to run the data collection directly from the command line using `ncu` in order to observe the application’s and the profiler’s command line output, e.g. `==ERROR==`` ``The`` ``application`` ``returned`` ``an`` ``error`` ``code`` ``(11)`

- **Failed to open/create lock file (path). Please check that this process has write permissions on it.**

  NVIDIA Nsight Compute failed to create or open the file `(path)` with write permissions. This file is used for inter-process [serialization](index.html#serialization). NVIDIA Nsight Compute does not remove this file after profiling by design. The error occurs if the file was created by a profiling process with permissions that prevent the current process from writing to this file, or if the current user can’t acquire this file for other reasons (e.g., certain Linux kernel security settings).

  The file is in the current temporary directory, i.e. `TMPDIR/nvidia/nsight_compute/lock.*` on Linux and `TMPDIR/nsight-compute-lock.*` on other platforms. On Windows, `TMPDIR` is the path returned by the Windows `GetTempPath` API function. On other platforms, it is the path supplied by the first environment variable in the list: `TMPDIR`, `TMP`, `TEMP`, `TEMPDIR`. If none of these are found, the default path is `/var/nvidia` on QNX and `/tmp` otherwise.

  Older versions of NVIDIA Nsight Compute did not set write permissions for all users on this file by default. As a result, running the tool on the same system with a different user might cause this error. This has been resolved since version 2020.2.1.

  The following workarounds can be used to solve this problem:

  - If it is otherwise ensured that no concurrent NVIDIA Nsight Compute instances are active on the same system, set `TMPDIR` to a different directory for which the current user has write permissions.

  - Ask the user that owns the file, or a system administrator, to remove it or add write permissions for all potential users.

- **Profiling failed because a driver resource was unavailable.**

  The error indicates that a required CUDA driver resource was unavailable during profiling. Most commonly, this means that NVIDIA Nsight Compute could not reserve the driver’s performance monitor, which is necessary for collecting most metrics.

  This can happen if another application has a concurrent reservation on this resource. Such applications can be e.g. [DCGM](https://developer.nvidia.com/dcgm), a client of [CUPTI’s Profiling API](https://developer.nvidia.com/cupti), [Nsight Graphics](https://developer.nvidia.com/nsight-graphics), or another instance of NVIDIA Nsight Compute without access to the same file system (see [serialization](index.html#serialization) for how this is prevented within the same file system).

  If you expect the problem to be caused by DCGM, consider using `dcgmi`` ``profile`` ``--pause` to stop its monitoring while profiling with NVIDIA Nsight Compute.

- **Could not deploy stock \* files to \***

  **Could not determine user home directory for section deployment.**

  An error occurred while trying to deploy stock section or rule files. By default, NVIDIA Nsight Compute tries to deploy these to a versioned directory in the user’s home directory (as identified by the `HOME` environment variable on Linux), e.g. `/home/user/Documents/NVIDIA`` ``Nsight`` ``Compute/<version>/Sections`.

  If the directory cannot be determined (e.g. because this environment variable is not pointing to a valid directory), or if there is an error while deploying the files (e.g. because the current process does not have write permissions on it), warning messages are shown and NVIDIA Nsight Compute falls back to using stock sections and rules from the installation directory.

  If you are in an environment where you consistently don’t have write access to the user’s home directory, consider populating this directory upfront using `ncu`` ``--section-folder-restore`, or by making `/home/user/Documents/NVIDIA`` ``Nsight`` ``Compute/<version>` a symlink to a writable directory.

- **ProxyJump SSH option is not working**

  NVIDIA Nsight Compute does not manage authentication or interactive prompts with the OpenSSH client launched when using the `ProxyJump` option. Therefore, to connect through an intermediate host for the first time, you will not be able to accept the intermediate host’s key. A simple way to pinpoint the cause of failures in this case is to open a terminal and use the OpenSSH client to connect to the remote target. Once that connection succeeds, NVIDIA Nsight Compute should be able to connect to the target, too.

- **SSH connection fails without trying to connect**

  If the connection fails without trying to connect, there may be a problem with the settings you entered into the *Start Activity Dialog*. Please make sure that the `IP/Host`` ``Name`, `User`` ``Name` and `Port` fields are correctly set.

- **SSH connections are still not working**

  The problem might come from NVIDIA Nsight Compute’s SSH client not finding a suitable host key algorithm to use which is supported by the remote server. You can force NVIDIA Nsight Compute to use a specific set of host key algorithms by setting the `HostKeyAlgorithms` option for the problematic host in your SSH configuration file. To list the supported host key algorithms for a remote target, you can use the `ssh-keyscan` utility which comes with the OpenSSH client.

- **Removing host keys from known hosts files**

  When connecting to a target machine, NVIDIA Nsight Compute tries to verify the target’s host key against the same local database as the OpenSSH client. If NVIDIA Nsight Compute find the host key is incorrect, it will inform you through a failure dialog. If you trust the key hash shown in the dialog, you can remove the previously saved key for that host by manually editing your known hosts database or using the `ssh-keygen`` ``-R`` ``<host>` command.

- **Qt initialization failed**

  **Failed to load Qt platform plugin**

  See [System Requirements](../ReleaseNotes/index.html#system-requirements) for Linux.

Notices

Notices

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

Information furnished is believed to be accurate and reliable. However, NVIDIA Corporation assumes no responsibility for the consequences of use of such information or for any infringement of patents or other rights of third parties that may result from its use. No license is granted by implication of otherwise under any patent rights of NVIDIA Corporation. Specifications mentioned in this publication are subject to change without notice. This publication supersedes and replaces all other information previously supplied. NVIDIA Corporation products are not authorized as critical components in life support devices or systems without express written approval of NVIDIA Corporation.

Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.


---


<!-- source: https://docs.nvidia.com/nsight-compute/NsightCompute/index.html -->
# 3. Nsight Compute

The User Guide for Nsight Compute.

## 3.1. Introduction

NVIDIA Nsight Compute (UI) User Manual: Detailed documentation on views, controls, and workflows.

### 3.1.1. Overview

This document is a user guide to the next-generation NVIDIA Nsight Compute profiling tools. NVIDIA Nsight Compute is an interactive kernel profiler for CUDA applications. It provides detailed performance metrics and API debugging via a user interface and command line tool. In addition, its baseline feature allows users to compare results within the tool. NVIDIA Nsight Compute provides a customizable and data-driven user interface and metric collection and can be extended with analysis scripts for post-processing results.

**Important Features**

- Interactive kernel profiler and API debugger

- Graphical profile report

- Result comparison across one or multiple reports within the tool

- Fast Data Collection

- UI and Command Line interface

- Fully customizable reports and analysis rules

## 3.2. Quickstart

The following sections provide brief step-by-step guides of how to setup and run NVIDIA Nsight Compute to collect profile information. All directories are relative to the base directory of NVIDIA Nsight Compute, unless specified otherwise.

The UI executable is called ncu-ui. A shortcut with this name is located in the base directory of the NVIDIA Nsight Compute installation. The actual executable is located in the folder `host\windows-desktop-win7-x64` on Windows or `host/linux-desktop-glibc_2_11_3-x64` on Linux. By default, when installing from a Linux `.run` file, NVIDIA Nsight Compute is located in `/usr/local/cuda-<cuda-version>/nsight-compute-<version>`. When installing from a `.deb` or `.rpm` package, it is located in `/opt/nvidia/nsight-compute/<version>` to be consistent with [Nsight Systems](https://developer.nvidia.com/nsight-systems). In Windows, the default path is `C:\Program`` ``Files\NVIDIA`` ``Corporation\Nsight`` ``Compute`` ``<version>`.

After starting NVIDIA Nsight Compute, by default the *Welcome Page* is opened. The *Start* section allows the user to start a new activity, open an existing report, create a new project or load an existing project. The *Continue* section provides links to recently opened reports and projects. The *Explore* section provides information about what is new in the latest release, as well as links to additional training. See [Environment](index.html#options-environment) on how to change the start-up action.



Welcome Page

### 3.2.1. Interactive Profile Activity

1.  **Launch the target application from NVIDIA Nsight Compute**

    When starting NVIDIA Nsight Compute, the *Welcome Page* will appear. Click on *Quick Launch* to open the *Connection* dialog. If the *Connection* dialog doesn’t appear, you can open it using the *Connect* button from the main toolbar, as long as you are not currently connected. Select your target platform on the left-hand side and your connection target (machine) from the *Connection* drop down. If you have your local target platform selected, `localhost` will become available as a connection. Use the + button to add a new connection target. Then, continue by filling in the details in the *Launch* tab. In the *Activity* panel, select the Interactive Profile activity to initiate a session that allows controlling the execution of the target application and selecting the kernels of interest interactively. Press *Launch* to start the session.

    

     

2.  **Launch the target application with tools instrumentation from the command line**

    The ncu can act as a simple wrapper that forces the target application to load the necessary libraries for tools instrumentation. The parameter `--mode=launch` specifies that the target application should be launched and suspended before the first instrumented API call. That way the application waits until we connect with the UI.

    ``` text
    $ ncu --mode=launch CuVectorAddDrv.exe
    ```

3.  **Launch NVIDIA Nsight Compute and connect to target application**

    

    Select the target machine at the top of the dialog to connect and update the list of attachable applications. By default, *localhost* is pre-selected if the target matches your current local platform. Select the *Attach* tab and the target application of interest and press *Attach*. Once connected, the layout of NVIDIA Nsight Compute changes into stepping mode that allows you to control the execution of any calls into the instrumented API. When connected, the *API Stream* window indicates that the target application waits before the very first API call.

    

     

4.  **Control application execution**

    Use the *API Stream* window to step the calls into the instrumented API. The dropdown at the top allows switching between different CPU threads of the application. *Step In* (F11), *Step Over* (F10), and *Step Out* (Shift + F11) are available from the *Debug* menu or the corresponding toolbar buttons. While stepping, function return values and function parameters are captured.

    

    Use *Resume* (F5) and *Pause* to allow the program to run freely. Freeze control is available to define the behavior of threads currently not in focus, i.e. selected in the thread drop down. By default, the *API Stream* stops on any API call that returns an error code. This can be toggled in the *Debug* menu by *Break On API Error*.

5.  **Isolate a kernel launch**

    To quickly isolate a kernel launch for profiling, use the *Run to Next Kernel* button in the toolbar of the *API Stream* window to jump to the next kernel launch. The execution will stop before the kernel launch is executed.

    

     

6.  **Profile a kernel launch**

    Once the execution of the target application is suspended at a kernel launch, additional actions become available in the UI. These actions are either available from the menu or from the toolbar. Please note that the actions are disabled, if the API stream is not at a qualifying state (not at a kernel launch or launching on an unsupported GPU). To profile, press *Profile Kernel* and wait until the result is shown in the [Profiler Report](index.html#profiler-report). Profiling progress is reported in the lower right corner status bar.

    Instead of manually selecting *Profile*, it is also possible to enable *Auto Profile* from the *Profile* menu. If enabled, each kernel matching the current kernel filter (if any) will be profiled using the current section configuration. This is especially useful if an application is to be profiled unattended, or the number of kernel launches to be profiled is very large. Sections can be enabled or disabled using the [Metric Selection](index.html#tool-window-sections-info) tool window.

    *Profile Series* allows to configure the collection of a set of profile results at once. Each result in the set is profiled with varying parameters. Series are useful to investigate the behavior of a kernel across a large set of parameters without the need to recompile and rerun the application many times.

For a detailed description of the options available in this activity, see [Interactive Profile Activity](index.html#connection-activity-interactive).

### 3.2.2. Non-Interactive Profile Activity

1.  **Launch the target application from NVIDIA Nsight Compute**

    When starting NVIDIA Nsight Compute, the *Welcome Page* will appear. Click on *Start Activity* to open the *Start Activity* dialog. If the *Start Activity* dialog doesn’t appear, you can open it using the *Start Activity* button from the main toolbar, as long as you are not currently connected. Select your target platform on the left-hand side and your localhost from the *Connection* drop down. Then, fill in the launch details. In the *Activity* panel, select the *Profile* activity to initiate a session that pre-configures the profile session and launches the command line profiler to collect the data. Provide the *Output File* name to enable starting the session with the *Launch* button.

    

2.  **Additional Launch Options**

    For more details on these options, see [Command Line Options](../NsightComputeCli/index.html#command-line-options). The options are grouped into tabs: The *Filter* tab exposes the options to specify which kernels should be profiled. Options include the kernel regex filter, the number of launches to skip, and the total number of launches to profile. The *Sections* tab allows you to select which sections should be collected for each kernel launch. Hover over a section to see its description as a tool-tip. To change the sections that are enabled by default, use the [Metric Selection](index.html#tool-window-sections-info) tool window. The *Sampling* tab allows you to configure sampling options for each kernel launch. The *Other* tab includes the option to collect NVTX information or custom metrics via the `--metrics` option.

    

For a detailed description of the options available in this activity, see [Profile Activity](index.html#connection-activity-non-interactive).

### 3.2.3. System Trace Activity

1.  **Launch the target application from NVIDIA Nsight Compute**

    When starting NVIDIA Nsight Compute, the *Welcome Page* will appear. Click on *Start Activity* to open the *Start Activity* dialog. If the *Start Activity* dialog doesn’t appear, you can open it using the *Start Activity* button from the main toolbar, as long as you are not currently connected. Select your local target platform on the left-hand side and your localhost from the *Connection* drop down. Then, fill in the launch details. In the *Activity* panel, select the *System Trace* activity to initiate a session with pre-configured settings. Press *Launch* to start the session.

    

2.  **Additional Launch Options**

    For more details on these options, see [System-Wide Profiling Options](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#linux-system-wide-profiling-options).

    

3.  Once the session is completed, the [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) report is opened in a new document. By default, the timeline view is shown. It provides detailed information of the activity of the CPU and GPUs and helps understanding the overall behavior and performance of application. Once a CUDA kernel is identified to be on the critical path and not meeting the performance expectations, right click on the kernel launch on timeline and select *Profile Kernel* from the context menu. A new [Start Activity](index.html#connection-dialog) opens up that is already preconfigured to profile the selected kernel launch. Proceed with optimizing the selected kernel using [Non-Interactive Profile Activity](index.html#quick-start-non-interactive)

    

### 3.2.4. Navigate the Report

1.  **Navigate the report**

    The profile report comes up by default on the *Summary* page. It shows an overview table to summarize all results in the report. It also shows rule information for the selected row.

    > 

    You can switch between different [Report Pages](index.html#profiler-report-pages) using the tab bar on the top-left of the report. You can also use *Ctrl + Shift + N* and *Ctrl + Shift + P* shortcut keys or corresponding toolbar button to navigate next and previous pages, respectively. A report can contain any number of results. The *Current* dropdown allows switching between the different results in a report.

    

2.  **Diffing multiple results**

    On the *Details* page, use the *Compare - Add Baseline* button for the current result to become the baseline all other results from this report and any other report opened in the same instance of NVIDIA Nsight Compute get compared to. When a baseline is set, every element on the Details page shows two values: The current value of the result in focus and the corresponding value of the baseline or the percentage of change from the corresponding baseline value.

    

    Use the *Clear Baselines* entry from the same group button, the Profile menu or the corresponding toolbar button to remove all baselines. For more information see [Baselines](index.html#baselines).

3.  **Following rules**

    On the *Details* page, many sections provide rules with valuable information on detected problems and optimization suggestions. Rules can be user-defined too. For more information, see the [Customization Guide](../CustomizationGuide/index.html#rule-system).

    

## 3.3. Start Activity Dialog

Use the *Start Activity Dialog* to launch and attach to applications on your local and remote platforms. Start by selecting the *Target Platform* for profiling. By default (and if supported) your local platform will be selected. Select the platform on which you would like to start the target application or connect to a running process.

[](../_images/connection-dialog.png)

When using a remote platform, you will be asked to select or create a *Connection* in the top drop down. To create a new connection, select *+* and enter your connection details. When using the local platform, *localhost* will be selected as the default and no further connection settings are required. You can still create or select a remote connection, if profiling will be on a remote system of the same platform. Use the *Device Information* button to view system and GPU information for the selected device.

Depending on your target platform, select either *Launch* or *Remote Launch* to launch an application for profiling on the target. Note that *Remote Launch* will only be available if supported on the target platform.

Fill in the following launch details for the application:

- **Application Executable:** Specifies the root application to launch. Note that this may not be the final application that you wish to profile. It can be a script or launcher that creates other processes.

- **Working Directory:** The directory in which the application will be launched.

- **Command Line Arguments:** Specify the arguments to pass to the application executable.

- **Environment:** The environment variables to set for the launched application.

Select *Attach* to attach the profiler to an application already running on the target platform. This application must have been started using another NVIDIA Nsight Compute CLI instance. The list will show all application processes running on the target system which can be attached. Select the refresh button to re-create this list.

Finally, select the *Activity* to be run on the target for the launched or attached application. Note that not all activities are necessarily compatible with all targets and connection options. Currently, the following activities exist:

- [Interactive Profile Activity](index.html#connection-activity-interactive)

- [Profile Activity](index.html#connection-activity-non-interactive)

- [System Trace Activity](index.html#quick-start-system-trace)

- [Occupancy Calculator](index.html#occupancy-calculator)

### 3.3.1. Remote Connections

Remote devices that support SSH can also be configured as a target in the *Start Activity Dialog*. To configure a remote device, ensure an SSH-capable *Target Platform* is selected, then press the *+* button. The following configuration dialog will be presented.



NVIDIA Nsight Compute supports both password and private key authentication methods. In this dialog, select the authentication method and enter the following information:

- **Password**

  - **IP/Host Name:** The IP address or host name of the target device.

  - **User Name:** The user name to be used for the SSH connection.

  - **Password:** The user password to be used for the SSH connection.

  - **Port:** The port to be used for the SSH connection. (The default value is 22)

  - **Deployment Directory:** The directory to use on the target device to deploy supporting files. The specified user must have write permissions to this location.

  - **Connection Name:** The name of the remote connection that will show up in the *Start Activity Dialog*. If not set, it will default to \<User\>@\<Host\>:\<Port\>.

- **Private Key**

  

  - **IP/Host Name:** The IP address or host name of the target device.

  - **User Name:** The user name to be used for the SSH connection.

  - **SSH Private Key:** The private key that is used to authenticate to SSH server.

  - **SSH Key Passphrase:** The passphrase for your private key.

  - **Port:** The port to be used for the SSH connection. (The default value is 22)

  - **Deployment Directory:** The directory to use on the target device to deploy supporting files. The specified user must have write permissions to this location.

  - **Connection Name:** The name of the remote connection that will show up in the *Start Activity Dialog*. If not set, it will default to \<User\>@\<Host\>:\<Port\>.

In addition to keyfiles specified by path and plain password authentication, NVIDIA Nsight Compute supports keyboard-interactive authentication, standard keyfile path searching and SSH agents.

You can also specify placeholders in the *Deployment Directory* field. These placeholders will be replaced with the actual values when the connection is used. The following placeholders are supported.

| Placeholder | Description |
|----|----|
| %C | User cache directory. Resolves to \$XDG_CACHE_HOME, if empty resolves to \$HOME/.cache |
| %E | User config directory. Resolves to \$XDG_CONFIG_HOME, if empty resolves to \$HOME/.config |
| %h | User home directory. Resolves to \$HOME, if empty resolves to /root |
| %H | Host name. Resolves to \$HOSTNAME, if empty resolves to host |
| %T | Temporary directory. Resolves to \$TMPDIR, if empty resolves to /tmp |
| %u | User name. Resolves to \$USER, if empty resolves to root |
| %S | State directory. Resolves to \$XDG_STATE_HOME, if empty resolves to \$HOME/.local/state |

Deployment Directory Placeholders {#id15 .table-no-stripes .docutils .align-default}

For example, if you specify the *Deployment Directory* as *%T/nsight-compute*, the actual directory used will be */tmp/nsight-compute* on the remote device.

When all information is entered, click the *Add* button to make use of this new connection.

When a remote connection is selected in the *Start Activity Dialog*, the *Application Executable* file browser will browse the remote file system using the configured SSH connection, allowing the user to select the target application on the remote device.

When an activity is launched on a remote device, the following steps are taken:

1.  The command line profiler and supporting files are copied into the *Deployment Directory* on the remote device. (Only files that do not exist or are out of date are copied.)

2.  Communication channels are opened to prepare for the traffic between the UI and the *Application Executable*.

    - For *Interactive Profile* activities, a *SOCKS proxy* is started on the host machine.

    - For *Non-Interactive Profile* activities, a remote forwarding channel is opened on the target machine to tunnel profiling information back to the host.

3.  The *Application Executable* is executed on the remote device.

    - For *Interactive Profile* activities, a connection is established to the remote application and the profiling session begins.

    - For *Non-Interactive Profile* activities, the remote application is executed under the command line profiler and the specified report file is generated.

4.  For non-interactive profiling activities, the generated report file is copied back to the host, and opened.

The progress of each of these steps is presented in the *Progress Log*.



Progress Log

Note that once either activity type has been launched remotely, the tools necessary for further profiling sessions can be found in the *Deployment Directory* on the remote device.

On Linux and Mac host platforms, NVIDIA Nsight Compute supports SSH remote profiling on target machines which are not directly addressable from the machine the UI is running on through the `ProxyJump` and `ProxyCommand` SSH options.

These options can be used to specify intermediate hosts to connect to or actual commands to run to obtain a socket connected to the SSH server on the target host and can be added to your SSH configuration file.

Note that for both options, NVIDIA Nsight Compute runs external commands and does not implement any mechanism to authenticate to the intermediate hosts using the credentials entered in the [Start Activity Dialog](index.html#connection-dialog). These credentials will only be used to authenticate to the final target in the chain of machines.

When using the `ProxyJump` option NVIDIA Nsight Compute uses the *OpenSSH client* to establish the connection to the intermediate hosts. This means that in order to use `ProxyJump` or `ProxyCommand`, a version of OpenSSH supporting these options must be installed on the host machine.

A common way to authenticate to the intermediate hosts in this case is to use a *SSH agent* and have it hold the private keys used for authentication.

Since the *OpenSSH SSH client* is used, you can also use the *SSH askpass* mechanism to handle these authentications in an interactive manner.

It might happen on slow networks that connections used for remote profiling through SSH time out. If this is the case, the `ConnectTimeout` option can be used to set the desired timeout value.

A known limitation of the remote profiling through SSH is that problems may arise if NVIDIA Nsight Compute tries to do remote profiling through *SSH* by connecting to the same machine it is running on. In this case, the workaround is to do local profiling through `localhost`.

For more information about available options for the *OpenSSH client* and the ecosystem of tools it can be used with for authentication refer to the official [manual pages](https://www.openssh.com/manual.html).

### 3.3.2. Interactive Profile Activity

The *Interactive Profile* activity allows you to initiate a session that controls the execution of the target application, similar to a debugger. You can step API calls and workloads (CUDA kernels), pause and resume, and interactively select the kernels of interest and which metrics to collect.

This activity does currently not support profiling or attaching to child processes.

- **Enable CPU Call Stack**

  Collect the CPU-sided Call Stack at the location of each profiled kernel launch.

- **CPU Call Stack Types**

  If “Enable CPU Call Stack” is set to “Yes”, the type(s) of call stack may be selected here.

- **Enable NVTX Support**

  Collect NVTX information provided by the application or its libraries. Required to support stepping to specific NVTX contexts.

- **Disable Profiling Start/Stop**

  Ignore calls to `cu(da)ProfilerStart` or `cu(da)ProfilerStop` made by the application.

- **Enable Profiling From Start**

  Enables profiling from the application start. Disabling this is useful if the application calls `cu(da)ProfilerStart` and kernels before the first call to this API should not be profiled. Note that disabling this does not prevent you from manually profiling kernels.

- **Cache Control**

  Control the behavior of the GPU caches during profiling. Allowed values: For *Flush All*, all GPU caches are flushed before each kernel replay iteration during profiling. While metric values in the execution environment of the application might be slightly different without invalidating the caches, this mode offers the most reproducible metric results across the replay passes and also across multiple runs of the target application.

  For *Flush None*, no GPU caches are flushed during profiling. This can improve performance and better replicates the application behavior if only a single kernel replay pass is necessary for metric collection. However, some metric results will vary depending on prior GPU work, and between replay iterations. This can lead to inconsistent and out-of-bounds metric values.

- **Clock Control**

  Control the behavior of the GPU clocks during profiling. Allowed values: For *Base*, GPC and memory clocks are locked to their respective base frequency during profiling. This has no impact on thermal throttling. For *Boost*, GPC and memory clocks are locked to their respective turbo boost frequency during profiling. If setting boost clock is not supported by the GPU or driver, falls back to locking clocks to ‘base’ if possible. For *Force-Boost*, GPC and memory clocks are locked to their respective turbo boost frequency during profiling. Supports Ampere+ on NVIDIA kernel mode driver 560 and Turing+ on driver 580 or newer. For *None*, no GPC or memory frequencies are changed during profiling.

- **Import Source**

  Enables permanently importing available source files into the report. Missing source files are searched in [Source Lookup](index.html#options-source-lookup) folders. Source information must be embedded in the executable, e.g. via the `-lineinfo` compiler option. Imported files are used in the *CUDA-C* view on the [Source Page](index.html#profiler-report-source-page).

&nbsp;

- **Graph Profiling**

  Set if CUDA graphs should be stepped and profiled as individual *Nodes* or as complete *Graphs*. See the [Profiling Guide](../ProfilingGuide/index.html#graph-profiling) for more information on this mode.

&nbsp;

- **Pipeline Boost State**

  Control the Tensor Core boost state. It is recommended to set the stable Tensor Core boosting for application profiling, ensuring predictive run-to-run performance. By default, the boost state is set to `Auto`, which attempts to set the stable boost state on supported platforms and drivers.

### 3.3.3. Profile Activity

The *Profile* activity provides a traditional, pre-configurable profiler. After configuring which kernels to profile, which metrics to collect, etc, the application is run under the profiler without interactive control. The activity completes once the application terminates. For applications that normally do not terminate on their own, e.g. interactive user interfaces, you can cancel the activity once all expected kernels are profiled.

This activity does not support attaching to processes previously launched via NVIDIA Nsight Compute. These processes will be shown grayed out in the *Attach* tab.

- **Output File**

  Path to report file where the collected profile should be stored. If not present, the report extension `.ncu-rep` is added automatically. The placeholder `%i` is supported for the filename component. It is replaced by a sequentially increasing number to create a unique filename. This maps to the `--export` command line option.

- **Force Overwrite**

  If set, existing report file are overwritten. This maps to the `--force-overwrite` command line option.

- **Target Processes**

  Select the processes you want to profile. In mode *Application Only*, only the root application process is profiled. In mode *all*, the root application process and all its child processes are profiled. This maps to the `--target-processes` command line option.

- **Replay Mode**

  Select the method for replaying kernel launches multiple times. In mode *Kernel*, individual kernel launches are replayed transparently during the single execution of the target application. In mode *Application*, the entire target application is relaunched multiple times. In each iteration, additional data for the target kernel launches is collected. Application replay requires the program execution to be deterministic. This maps to the `--replay-mode` command line option. See the [Profiling Guide](../ProfilingGuide/index.html#kernel-replay) for more details on the replay modes.

&nbsp;

- **Graph Profiling**

  Set if CUDA graphs should be profiled as individual *Nodes* or as complete *Graphs*.

&nbsp;

- **Additional Options**

  All remaining options map to their command line profiler equivalents. See the [Command Line Options](../NsightComputeCli/index.html#command-line-options) for details.

### 3.3.4. Reset

Entries in the connection dialog are saved as part of the current [project](index.html#projects). When working in a custom project, simply close the project to reset the dialog.

When not working in a custom project, entries are stored as part of the *default project*. You can delete all information from the default project by closing NVIDIA Nsight Compute and then [deleting the project file from disk](index.html#projects).

## 3.4. Main Menu and Toolbar

Information on the main menu and toolbar.



### 3.4.1. Main Menu

- File

  - **New Project** Create new profiling [Projects](index.html#projects) with the [New Project Dialog](index.html#projects-dialog).

  - **Open Project** Open an existing profiling project.

  - **Recent Projects** Open an existing profiling project from the list of recently used projects.

  - **Save Project** Save the current profiling project.

  - **Save Project As** Save the current profiling project with a new filename.

  - **Close Project** Close the current profiling project.

  - **New File** Create a new file.

  - **Open File** Open an existing file.

  - **Open Remote File**

    Download an existing file from a remote host and open it locally. The opened file will only exist in memory and will not be written to the local machine’s disk unless the user explicitly saves it. For more information concerning the selection of a remote host to download the file from, see the section about [Remote Connections](index.html#remote-connections).

    Only a subset of file types that are supported locally can be opened from a remote target. The following table lists file types that can be opened remotely.

    | Extensions | Description | Supported |
    |----|----|----|
    | ncu-rep | Nsight Compute Profiler Report | Yes |
    | ncu-repz | Nsight Compute Profiler Report (zstd compressed) | Yes |
    | ncu-occ | Occupancy Calculator File | Yes |
    | ncu-bvh | OptiX AS Viewer File | Yes (except on macOS) |
    | section | Section Description | No |
    | cubin | Cubin File | No |
    | cuh,h,hpp | Header File | No |
    | c,cpp,cu | Source File | No |
    | txt | Text file | No |
    | nsight-cuprof-report | Nsight Compute Profiler Report (legacy) | Yes |

    Remote File Type Support {#id16 .table-no-stripes .docutils .align-default}

  - **Save** Save the current file

  - **Save As** Save a copy of the current file with a different name or type or in a different location.

  - **Save All Files** Save all open files.

  - **Close** Close the current file.

  - **Close All Files** Close all open files.

  - **Recent Files** Open an existing file from the list of recently used files.

  - **Exit** Exit Nsight Compute.

- Connection

  - **Start Activity** Open the [Start Activity Dialog](index.html#connection-dialog) to launch or attach to a target application. Disabled when already connected.

  - **Disconnect** Disconnect from the current target application, allows the application to continue normally and potentially re-attach.

  - **Terminate** Disconnect from and terminate the current target application immediately.

  - **Device Information** Open the [Device Information](index.html#tool-window-device-information) tool window to view system and GPU information for the selected device.

- Debug

  - **Pause** Pause the target application at the next intercepted API call or launch.

  - **Resume** Resume the target application.

  - **Step In** Step into the current API call or launch to the next nested call, if any, or the subsequent API call, otherwise.

  - **Step Over** Step over the current API call or launch and suspend at the next, non-nested API call or launch.

  - **Step Out** Step out of the current nested API call or launch to the next, non-parent API call or launch one level above.

  - **Freeze API**

    When disabled, all CPU threads are enabled and continue to run during stepping or resume, and all threads stop as soon as at least one thread arrives at the next API call or launch. This also means that during stepping or resume the currently selected thread might change as the old selected thread makes no forward progress and the API Stream automatically switches to the thread with a new API call or launch. When enabled, only the currently selected CPU thread is enabled. All other threads are disabled and blocked.

    Stepping now completes if the current thread arrives at the next API call or launch. The selected thread never changes. However, if the selected thread does not call any further API calls or waits at a barrier for another thread to make progress, stepping may not complete and hang indefinitely. In this case, pause, select another thread, and continue stepping until the original thread is unblocked. In this mode, only the selected thread will ever make forward progress.

  - **Break On API Error** When enabled, during resume or stepping, execution is suspended as soon as an API call returns an error code. Note that some non-zero return codes are expected and not necessarily errors. In addition, certain errors may be ignored if they don’t happen on a nested API call, as the top-level call may still handle them.

  - **Run to Next Kernel** See [API Stream](index.html#tool-window-api-stream) tool window.

  - **Run to Next API Call** See [API Stream](index.html#tool-window-api-stream) tool window.

  - **Run to Next Range Start** See [API Stream](index.html#tool-window-api-stream) tool window.

  - **Run to Next Range End** See [API Stream](index.html#tool-window-api-stream) tool window.

  - **API Statistics** Opens the [API Statistics](index.html#tool-window-api-statistics) tool window

  - **API Stream** Opens the [API Stream](index.html#tool-window-api-stream) tool window

  - **Resources** Opens the [Resources](index.html#tool-window-resources) tool window

  - **NVTX** Opens the [NVTX](index.html#tool-window-nvtx) tool window

- Profile

  - **Profile Kernel** When suspended at a kernel launch, select the profile using the current configuration.

  - **Profile Series** When suspended at a kernel launch, open the Profile Series configuration dialog to setup and collect a series of profile results.

  - **Auto Profile** Enable or disable auto profiling. If enabled, each kernel matching the current kernel filter (if any) will be profiled using the current section configuration.

  - **Baselines** Opens the [Baselines](index.html#tool-window-baselines) tool window.

  - **Clear Baselines** Clear all current baselines.

  - **Import Source** Permanently import resolved source files into the report. Existing content may be overwritten.

  - **Section/Rules Info** Opens the [Metric Selection](index.html#tool-window-sections-info) tool window.

  - **Function Stats** Opens the [Function Stats](index.html#tool-window-function-stats) tool window.

- Tools

  - **Project Explorer** Opens the [Project Explorer](index.html#projects-explorer) tool window.

  - **Output Messages** Opens the Output Messages tool window.

  - **Options** Opens the [Options](index.html#options) dialog.

  - **Search** Opens the [Search](index.html#tool-window-search) tool window.

- Window

  - **Save Window Layout** Allows you to specify a name for the current layout. The layouts are saved to a Layouts folder in the documents directory as named “.nvlayout” files.

  - **Apply Window Layout** Once you have saved a layout, you can restore them by using the “Apply Window Layout” menu entry. Simply select the entry from sub-menu you want to apply.

  - **Manage Window Layout** Allows you to delete or rename old layouts.

  - **Restore Default Layout** Restore views to their original size and position.

  - **Show Welcome Page** Opens the [Welcome Page](index.html#quick-start__fig-welcome-page).

- Help

  - **Documentation** Opens the latest documentation for NVIDIA Nsight Compute online.

  - **Documentation (local)** Opens the local HTML documentation for NVIDIA Nsight Compute that has shipped with the tool.

  - **Check For Updates** Checks online if a newer version of NVIDIA Nsight Compute is available for download.

  - **Reset Application Data** Reset all NVIDIA Nsight Compute configuration data saved on disk, including option settings, default paths, recent project references etc. This will not delete saved reports.

  - **Send Feedback** Opens a dialog that allows you to send bug reports and suggestions for features. Optionally, the feedback includes basic system information, screenshots, or additional files (such as profile reports).

  - **About** Opens the About dialog with information about the version of NVIDIA Nsight Compute.

### 3.4.2. Main Toolbar

The main toolbar shows commonly used operations from the main menu. See [Main Menu](index.html#main-menu) for their description.

### 3.4.3. Status Banners

Status banners are used to display important messages, such as profiler errors. The message can be dismissed by clicking the ‘X’ button. The number of banners shown at the same time is limited and old messages can get dismissed automatically if new ones appear. Use the *Output Messages* window to see the complete message history.



## 3.5. Tool Windows

### 3.5.1. API Statistics

The *API Statistics* window is available when NVIDIA Nsight Compute is connected to a target application. It opens by default as soon as the connection is established. It can be re-opened using *Debug \> API Statistics* from the main menu.



Whenever the target application is suspended, it shows a summary of tracked API calls with some statistical information, such as the number of calls, their total, average, minimum and maximum duration. Note that this view cannot be used as a replacement for [Nsight Systems](https://developer.nvidia.com/nsight-systems) when trying to optimize CPU performance of your application.

The *Reset* button deletes all statistics collected to the current point and starts a new collection. Use the *Export to CSV* button to export the current statistics to a CSV file.

### 3.5.2. API Stream

The *API Stream* window is available when NVIDIA Nsight Compute is connected to a target application. It opens by default as soon as the connection is established. It can be re-opened using *Debug \> API Stream* from the main menu.



Whenever the target application is suspended, the window shows the history of API calls and traced kernel launches. The currently suspended API call or kernel launch (activity) is marked with a yellow arrow. If the suspension is at a subcall, the parent call is marked with a green arrow. The API call or kernel is suspended before being executed.

For each activity, further information is shown such as the kernel name or the function parameters (*Func Parameters*) and return value (*Func Return*). Note that the function return value will only become available once you step out or over the API call.

Use the *Current Thread* dropdown to switch between the active threads. The dropdown shows the thread ID followed by the current API name. One of several options can be chosen in the trigger dropdown, which are executed by the adjacent *\>\>* button. *Run to Next Kernel* resumes execution until the next kernel launch is found in any enabled thread. *Run to Next API Call* resumes execution until the next API call matching *Next Trigger* is found in any enabled thread. *Run to Next Range Start* resumes execution until the next start of an active profiler range is found. Profiler ranges are defined by using the `cu(da)ProfilerStart/Stop` API calls. *Run to Next Range Stop* resumes execution until the next stop of an active profiler range is found. The *API Level* dropdown changes which API levels are shown in the stream. The *Export to CSV* button exports the currently visible stream to a CSV file.

### 3.5.3. Baselines

The *Baselines* tool window can be opened by clicking the *Baselines* entry in the *Profile* menu. It provides a centralized place from which to manage configured baselines. (Refer to [Baselines](index.html#baselines), for information on how to create baselines from profile results.)



The baseline visibility can be controlled by clicking on the check box in a table row. When the check box is checked, the baseline will be visible in the summary header as well as all graphs in all sections. When unchecked the baseline will be hidden and will not contribute to metric difference calculations.

The baseline color can be changed by double-clicking on the color swatch in the table row. The color dialog which is opened provides the ability to choose an arbitrary color as well as offers a palette of predefined colors associated with the stock baseline color rotation.

The baseline name can be changed by double-clicking on the *Name* column in the table row. The name must not be empty and must be less than the *Maximum Baseline Name Length* as specified in the options dialog.

The z-order of a selected baseline can be changed by clicking the *Move Baseline Up* and *Move Baseline Down* buttons in the tool bar. When a baseline is moved up or down its new position will be reflected in the report header as well as in each graph. Currently, only one baseline may be moved at a time.

The selected baselines may be removed by clicking on the *Clear Selected Baselines* button in the tool bar. All baselines can be removed at once by clicking on the *Clear All Baselines* button, from either the global tool bar or the tool window tool bar.

The configured baselines can be saved to a file by clicking on the *Save Baselines* button in the tool bar. By default baseline files use the `.ncu-bln` extension. Baseline files can be opened locally and/or shared with other users.

Baseline information can be loaded by clicking on the *Load Baselines* button in the tool bar. When a baseline file is loaded, currently configured baselines will be replaced. A dialog will be presented to the user to confirm this operation when necessary.

Differences between the current result and the baselines can be visualized with graphical bars for metrics in Details page section headers. Use the *Difference Bars* drop down to select the visualization mode. Bars are extending from left to right and have a fixed maximum.

### 3.5.4. Metric Details

The *Metric Details* tool window can be opened using the *Metric Details* entry in the *Profile* menu or the respective tool bar button. When a report and the tool window are open, a metric can be selected in the report to display additional information and its value for the current result in the tool window. It also contains a search bar to look up metrics in the focused report’s current result.



Report metrics can be selected on the [Details Page](index.html#profiler-report-details-page) or on the [Raw Page](index.html#profiler-report-raw-page). The window will show basic information (name, unit and raw value of the metric) as well as additional information, such as its extended description. Note that all shown values are for the respective metric in the current result, even if a different result in e.g. the Raw Page is selected.

For results collected for [Green Contexts](../ProfilingGuide/index.html#cuda-green-contexts) applications, the *Metric Details* tool window also shows an additional row with the *Attribution* level of the selected performance metric (e.g., Context or Green Context, if applicable).

The search bar can be used to open metrics in the focused report. It shows available matches as you type. The entered string must match from the start of the metric name.

By default, selecting or searching for a new metric updates the current *Default Tab*. You can click the pin button located in the upper-left corner of the tab control to create a copy of the default tab, unless the same metric is already pinned. This makes it possible to save multiple tabs and quickly switch between them to compare values.

Some metrics contain [Instance Values](../ProfilingGuide/index.html#metrics-structure). When available, they are listed in the tool window. Instance values can have a *Correlation ID* that allows correlating the individual value with its associated entity, e.g. a function address or instruction name.

For metrics collected with [PM sampling](../ProfilingGuide/index.html#pm-sampling), the correlation ID is the GPU timestamp in nanoseconds. It is shown relative to the first timestamp for this metric. Note that the instance values for PM sampling metrics are currently not [context-switched](../ProfilingGuide/index.html#ctx-switch-trace) , even if this is enabled in the corresponding timeline UI.

### 3.5.5. Launch Details

The *Launch Details* tool window can be opened using the *Launch Details* entry in the *Profile* menu or the respective tool bar button. When a result containing multiple sub-launches is selected and this tool window is open, it will display information about each sub-launch contained in the result.

[](../_images/tool-window-launch-details.png)

This tool window is split into two sections:

- a header displaying information applying to the result as a whole

- a body displaying information specific to the viewed sub-launch

#### Header

On the left side of its header, this tool window displays the selected result’s name and the number of sub-launches it is comprised of.

The right side contains a combo box that allows selection of the sub-launch the body should represent. Each element of the combo box contains an index for the sub-launch as well as the name of the function that it launched if available.

#### Body

The body of this tool window displays a table with sub-launch-specific metrics. This table has four columns:

- *Metric Name*: the name of the metric

- *Metric Unit*: the unit for metric values

- *Instance Value*: the value of this metric for the selected sub-launch

- *Aggregate Value*: the aggregate value for this metric over all sub-launches in the selected result

### 3.5.6. Function Stats

The *Function Stats* tool window presents a hierarchical tree view of all functions in your profile result with top stalls for each function and source line. This window is connected with the PM Sampling timeline and automatically updates to show data for the selected time range in the timeline.

#### Accessing Function Stats

The Function Stats tool window can be opened using:  
- The *Function Stats* entry in the *Profile* menu. See [Main Menu](index.html#main-menu) for details.

- The main tool bar button

#### Basic Workflow

1.  Profile with PM Warp Sampling enabled (default: on).

2.  Open the tool window using the **Function Stats** tool bar button.

3.  Expand a function to see per-source-line statistics.

4.  Sort by a column (default is sorted by **All Samples**).

5.  Click a function or source line to navigate to it on the Source page (if available).

Note

The source view is not updated with the warp sampling data collected with PM sampling.



The Function Stats tool window

#### Understanding the Function Stats Tree

The view displays a hierarchical tree with two levels:

- **Function entries** (top-level): Each function appears as a parent row showing aggregate statistics across all source lines.

- **Source line entries** (children): Expand a function to see per-source-line breakdown in `filename:line` format. Up to 10 source lines with the highest sample counts are shown. If a selection results in more than 10 source lines, the remaining ones are aggregated into an **Other** entry.

The columns include:

**Name**: The function name (for top-level entries) or source location in `filename:line` format (for child entries). Click to navigate to the corresponding location on the Source page.

**Samples (All)**: Reports the samples collected for this entry as % of total samples collected. This also includes all the stall reasons as a stacked bar chart.

**Samples (Not Issued)**: Reports the samples collected for this entry as % of total samples collected where warps were not issued. This also includes all the stall reasons as a stacked bar chart.

**Top Stall \#1**: Reports the stall reason with the highest incidence for the entry.

**Top Stall \#2**: Reports the stall reason with the second highest incidence for the entry.

**Top Stall \#3**: Reports the stall reason with the third highest incidence for the entry.

Note

By default, the tree is **sorted by All Samples in decreasing order**. When there is only one function in the view, it is automatically expanded to show all source lines.



The Function Stats tree view with source line links and tooltip

See the *Warp Stall Reasons* tables in the [Metrics Reference](../ProfilingGuide/index.html#warp-stall-reasons) for a description of the individual warp scheduler states.

#### Data Source

- Collection is performed by PM Sampling with warp-state sampling enabled (**default**).

- The shipped `PmSampling.section` file comes pre-configured with PM Sampling and Warp State Sampling metrics required to enable this window.

- At least one **PM Sampling** metric is required with the attribute `PmWarpSampling` declared (`Visible` or `Hidden`) to enable PM Warp Sampling data collection for this window. Use:

  - `PmWarpSampling:`` ``Visible` — show the metric with the timeline UI.

  - `PmWarpSampling:`` ``Hidden` — hide the metric in the timeline UI but still collected.

- The feature is **enabled by default** on all GPUs GA10X and newer. Disable globally with `--disable-pm-warp-sampling`.

- Only **one** PM Sampling pass with warp-state sampling is supported per profile session.

- **Note:** `warpsampling:` metrics alone do **not** populate the per-function view; at least one **PM Sampling** metric with the attribute `PmWarpSampling` must be present.

#### Metric Naming

Warp Stall Sampling metrics are the **same underlying source counter metrics**; they are requested by prefixing the counter name with `warpsampling:` to indicate collection via the PM Sampling path (**PM Warp Sampling**) and to render them on a timeline. Example: - `warpsampling:smsp__pcsamp_warps_issue_stalled_barrier`

In addition, a new group of **Warp Sampling** metrics that can be collected in this mode is available: - group:smsp\_\_pmwarpsamp_warp_stall_reasons

### 3.5.7. NVTX

The *NVTX* window is available when NVIDIA Nsight Compute is connected to a target application. If closed, it can be re-opened using *Debug \> NVTX* from the main menu. Whenever the target application is suspended, the window shows the state of all active NVTX domains and ranges in the currently selected thread. Note that [NVTX](https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/) information is only tracked if the launching command line profiler instance was started with `--nvtx` or NVTX was enabled in the NVIDIA Nsight Compute launch dialog.



Use the *Current Thread* dropdown in the [API Stream](index.html#tool-window-api-stream) window to change the currently selected thread. NVIDIA Nsight Compute supports NVTX named resources, such as threads, CUDA devices, CUDA contexts, etc. If a resource is named using NVTX, the appropriate UI elements will be updated.



### 3.5.8. CPU Call Stack

The *CPU Call Stack* window is available when NVIDIA Nsight Compute is connected to a target application. If closed, it can be re-opened using *Debug \> CPU Call Stack* from the main menu. Whenever the target application is suspended, the window shows all enabled CPU call stacks for the currently selected thread.



Use the *Call Stack Type* dropdown menu to switch between stack types in case multiple stack types were enabled (e.g., *Native*, *Python*). Note that Python call stack collection requires CPython version 3.9 or later.

### 3.5.9. Resources

The *Resources* window is available when NVIDIA Nsight Compute is connected to a target application. It shows information about the currently known resources, such as CUDA devices, CUDA streams or kernels. The window is updated every time the target application is suspended. If closed, it can be re-opened using *Debug \> Resources* from the main menu.



Using the dropdown on the top, different views can be selected, where each view is specific to one kind of resource (context, stream, kernel, …). The *Filter* edit allows you to create filter expressions using the column headers of the currently selected resource.

The resource table shows all information for each resource instance. Each instance has a unique ID, the *API Call ID* when this resource was created, its handle, associated handles, and further parameters. When a resource is destroyed, it is removed from its table.

#### Memory Allocations

When using the asynchronous malloc/free APIs, the resource view for *Memory Allocation* will also include the memory objects created in this manner. These memory objects have a non-zero memory pool handle. The *Mode* column will indicate which code path was taken during the allocation of the corresponding object. The modes are:

- REUSE_STREAM_SUBPOOL: The memory object was allocated in memory that was previously freed. The memory was backed by the memory pool set as current for the stream on which the allocation was made.

- USE_EXISTING_POOL_MEMORY: The memory object was allocated in memory that was previously freed. The memory is backed by the default memory pool of the stream on which the allocation was made.

- REUSE_EVENT_DEPENDENCIES: The memory object was allocated in memory that was previously freed in another stream of the same context. A stream ordering dependency of the allocating stream on the free action existed. Cuda events and null stream interactions can create the required stream ordered dependencies.

- REUSE_OPPORTUNISTIC: The memory object was allocated in memory that was previously freed in another stream of the same context. However, no dependency between the free and allocation existed. This mode requires that the free be already committed at the time the allocation is requested. Changes in execution behavior might result in different modes for multiple runs of the application.

- REUSE_INTERNAL_DEPENDENCIES: The memory object was allocated in memory that was previously freed in another stream of the same context. New internal stream dependencies may have been added in order to establish the stream ordering required to reuse a piece of memory previously released.

- REQUEST_NEW_ALLOCATION: New memory had to be allocated for this memory object as no viable reusable pool memory was found. The allocation performance is comparable to using the non-asynchronous malloc/free APIs.

#### Graphviz DOT and SVG exports

Some of the shown *Resources* can also be exported to *GraphViz DOT* or SVG\* files using the `Export`` ``to`` ``GraphViz` or `Export`` ``to`` ``SVG` buttons.

When exporting *OptiX traversable handles*, the traversable graph node types will be encoded using shapes and colors as described in the following table.

[TABLE]

OptiX Traversable Graph Node Types {#id17 .table-no-stripes .colwidths-given .docutils .align-default}

### 3.5.10. CUDA Graph Viewer

The *CUDA Graph Viewer* tool window provides real-time visualization and inspection of CUDA Graphs. Access it through the *CUDA Graph Viewer* entry in the *Profile* menu or via the toolbar button.

The viewer supports two primary workflows:

**Interactive Profiling**  
When profiling CUDA Graph workloads in an interactive session, the viewer automatically opens and displays graphs as the target application creates them. Each time execution is suspended, the viewer updates to show the current state of all active CUDA Graphs, including any structural changes or execution progress.

**Profile Report Analysis**  
When opening a profile report containing CUDA Graph data, the viewer automatically opens and displays the associated graphs. You can navigate between graphs by clicking on *Kernel* or *Graph* results in the [Summary Page](index.html#profiler-report-summary-page) or [Raw Page](index.html#profiler-report-raw-page) tables. The viewer highlights the specific graph node associated with each selected result.

By default, the viewer opens automatically upon graph creation. To disable this behavior, navigate to *Tools \> Options \> Profile \> CUDA Graph Viewer \> Auto-Open Graph Viewer* and set it to *No*.



Use the dropdown menu to select which CUDA Graph to view when multiple graphs are active. The window provides two tabs for examining CUDA Graphs:

- **Source Graph**: Displays the original graph structure as defined in the application source code before instantiation. This view reflects the logical organization of nodes and dependencies as specified by the developer.

- **Instantiated Graph**: Shows the graph as it exists after instantiation and optimization by the CUDA runtime. This view may differ from the source graph due to runtime optimizations, node merging, or dependency resolution.

Note that the graph layouts are optimized for interactive visualization and may differ from those generated by CUDA API functions such as `cudaGraphDebugDotPrint` or exported as [DOT or SVG](index.html#graphviz-dot-and-svg-exports) from the *Resources* window. The viewer uses hierarchical layout algorithms to improve readability and highlight execution flow.

You can toggle the visibility of node and edge labels using the *Show/Hide Labels* button. Hover over nodes, edges, or the graph canvas to view detailed tooltips.

**Execution Tracking**

The viewer provides dynamic execution tracking during both graph construction and execution:

- When a source graph is being built, the *Source Graph* tab automatically shows the graph in its current state. As nodes are added through CUDA API calls, the viewer updates in real-time to highlight newly created nodes and their connections.

- When an instantiated graph is launched, the *Instantiated Graph* tab automatically displays the graph execution state. As you step through the application, the viewer highlights nodes currently being executed and visually distinguishes completed nodes from pending ones. Execution flow through conditional graph and child graph nodes is clearly indicated.

### 3.5.11. Device Information

The *Device Information* tool window displays system and GPU information for the selected device connection. Access it through the *Device Information* button (info icon) in the [Start Activity Dialog](index.html#connection-dialog) or via *Connection \> Device Information* from the main menu.

[](../_images/connection-menu.png)

Access Device Information from the Connection menu.

The window provides two tabs for viewing device details:

**System**  
Displays operating system information including OS name, CPU architecture, hostname, IP addresses, and collection timestamp.

**GPU**  
Shows GPU-related information including driver version, CUDA version, and detailed properties for each detected GPU.

[](../_images/tool-window-device-information.png)

The Device Information window showing system and GPU information tabs.

Use the connection dropdown at the top to select a different device. Information is automatically collected when you select a device. Use the refresh button to manually update the information for the currently selected device.

The Device Information window supports both local and remote device connections. For remote devices, the system automatically deploys a collector binary via SSH and retrieves the information. Remote Windows targets are not supported.

### 3.5.12. Search

The *Search* tool window can be opened using the *Search* entry in the *Tools* menu, or from the tool bar’s search bar. It can be used to search for terms throughout Nsight Compute.

Your queries can also be searched in the online developer forums, providing you with additional information and help.



Enter your query into the tool menu’s or window’s search bar and press *Enter* or use the *Start search* button. Available sources are checked for matches in the background and results are displayed in the tool window’s table. While a search is running, you can press the *Cancel search* button to stop it. Previous searches are available from a history dropdown that is shown when clicking the tool window’s search bar with the mouse. Select a history entry to replace the current search bar content with it.



Each result is associated with the original source. While some results are plain text, others are links that can be clicked to jump to the location and see them in context. Below is a (non-comprehensive) list of supported sources. The list of available sources may be extended in the future.

- Local **Documentation**

- **Knowledgebase** entries

- **Metrics** in the focused report

- Nsight Compute UI **Options**

- Loaded **Sections**

- Online **Developer Forums** entries

### 3.5.13. Metric Selection

The *Metric Selection* window can be opened from the main menu using *Profile \> Metric Selection*. It tracks all metric sets, sections and rules currently loaded in NVIDIA Nsight Compute, independent from a specific connection or report. The directory to load those files from can be configured in the [Profile](index.html#options-profile) options dialog. It is used to inspect available sets, sections and rules, as well as to configure which should be collected, and which rules should be applied. You can also specify a comma separated list of individual metrics, that should be collected. The window has two views, which can be selected using the dropdown in its header.

The **Metric Sets** view shows all available metric sets. Each set is associated with a number of metrics sections. You can choose a set appropriate to the level of detail for which you want to collect performance metrics. Sets which collect more detailed information normally incur higher runtime overhead during profiling.



When enabling a set in this view, the associated metric sections are enabled in the *Metric Sections/Rules* view. When disabling a set in this view, the associated sections in the *Metric Sections/Rules* view are disabled. If no set is enabled, or if sections are manually enabled/disabled in the *Metric Sections/Rules* view, the \<*custom*\> entry is marked active to represent that no section set is currently enabled. Note that the *basic* set is enabled by default.

Whenever a kernel is profiled manually, or when auto-profiling is enabled, only sections enabled in the **Metric Sections/Rules** view and individual metrics specified in input box are collected. Similarly, whenever rules are applied, only rules enabled in this view are active.



The enabled states of sections and rules are persisted across NVIDIA Nsight Compute launches. The *Reload* button reloads all sections and rules from disk again. If a new section or rule is found, it will be enabled if possible. If any errors occur while loading a rule, they will be listed in an extra entry with a warning icon and a description of the error.

Use the *Enable All* and *Disable All* checkboxes to enable or disable all sections and rules at once. The Filter text box can be used to filter what is currently shown in the view. It does not alter activation of any entry.

The table shows sections and rules with their activation status, their relationship and further parameters, such as associated metrics or the original file on disk. Rules associated with a section are shown as children of their section entry. Rules independent of any section are shown under an additional *Independent Rules* entry.

Clicking an entry in the table’s *Filename* column opens this file as a document. It can be edited and saved (ctrl + s) directly in NVIDIA Nsight Compute. After editing the file, *Reload* must be selected to apply those changes. Document also supports text search (ctrl + f), zoom in (ctrl + mouse scroll down), zoom out (ctrl + mouse scroll up) functionalities.

When a section or rule file is modified, the entry in the *State* column will show *User Modified* to reflect that it has been modified from its default state. When a *User Modified* row is selected, the *Restore* button will be enabled. Clicking the Restore button will restore the entry to its default state and automatically *Reload* the sections and rules.

Similarly, when a stock section or rule file is removed from the configured *Sections Directory* (specified in the [Profile](index.html#options-profile) options dialog), the *State* column will show *User Deleted*. *User Deleted* files can also be restored using the *Restore* button.

Section and rule files that are created by the user (and not shipped with NVIDIA Nsight Compute) will show up as *User Created* in the *state column*.

See the [Sections and Rules](../ProfilingGuide/index.html#sections-and-rules) for the list of default sections for NVIDIA Nsight Compute.

## 3.6. Profiler Report

The profiler report contains all the information collected during profiling for each kernel launch. In the user interface, it consists of a header with general information, as well as controls to switch between report pages or individual collected launches.

### 3.6.1. Header



The top of the report shows a table with information about the selected profile result (as *Current*) and potentially additional [baselines](index.html#tool-window-baselines). For many values in this table, tooltips provide additional information or data, e.g., the tooltip of the column *Attributes* provides additional information about the context type and resources used for the launch.

The *Result* dropdown can be used to switch between all collected kernel launches. The information displayed in each page commonly represents the selected launch instance. On some pages (e.g. *Raw*), information for all launches is shown and the selected instance is highlighted. You can type in this dropdown to quickly filter and find a kernel launch.

The *Apply Filters* button opens the filter dialog. You can use more than one filter to narrow down your results. On the filter dialog, enter your filter parameters and press OK button. The *Launch* dropdown, [Summary Page](index.html#profiler-report-summary-page) table, and [Raw Page](index.html#profiler-report-raw-page) table will be filtered accordingly. Select the arrow dropdown to access the *Clear Filters* button, which removes all filters.



Filter Dialog

Underneath the current and baseline results are the tabs for switching between the report pages. The pages themselves are explained in detail in the [next section](index.html#profiler-report-pages).

Each group button to the right of the page tabs opens a context menu that features related actions. Some actions may be enabled only when the related report page is selected.

- Compare

  - **Add Baseline** promotes the current result in focus to become the baseline of all other results from this report and any other report opened in the same instance of NVIDIA Nsight Compute.

  - **Clear Baselines** removes all currently active baselines. You may also use the [Baselines](index.html#tool-window-baselines) tool window to manage baseline for comparison.

  - **Source Comparison** navigates to the [Source Comparison](index.html#source-comparison) document in case at least two profile results are available for comparison.

- Tools

  - **Occupancy Calculator** opens the [Occupancy Calculator](index.html#occupancy-calculator) in a new document.

  - **Metric Details Windows** opens the [Metric Details](index.html#tool-window-metric-details) tool window. When the window is open and a metric is selected elsewhere in the report, it shows detailed information about it.

  - **Launch Details Windows** opens the [Launch Details](index.html#tool-window-launch-details) tool window. When the window is open and a result containing multiple sub-launches is selected, it displays information about each sub-launch in the result.

  - **CUDA Graph Viewer** opens the [CUDA Graph Viewer](index.html#tool-window-cuda-graph-viewer) tool window. When the window is open and a result containing a CUDA Graph is selected, it displays the graph in the viewer.

- View

  - **Show/Hide Rules Output** toggles the visibility of rule results.

  - **Show/Hide Section Descriptions** toggles the visibility of section descriptions on the Details and Session pages.

  - **Show/Hide Green Context Markers** toggles the visibility of markers for *attributable* Green Context metrics.\`

  - **Expand Sections** expands all sections to show their body contents, not only header and rule output. Note that sections may have multiple bodies and the visible one can be chosen using the dropdown in the section header.

  - **Collapse Sections** collapses all sections to show only their header and rule output.

- Export

  - **Copy as Image** - Copies the contents of the page to the clipboard as an image.

  - **Save as Image** - Saves the contents of the page to a file as an image.

  - **Save as PDF** - Saves the contents of the page to a file as a PDF.

  - **Export to CSV** - Exports the contents of the page to CSV format.

- More (three bars icon)

  - **Apply Rules** applies all rules available for this report. If rules had been applied previously, those results will be replaced. By default, rules are applied immediately once the kernel launch has been profiled. This can be changed in the options under *Tools \> Options \> Profile \> Report UI \> Apply Applicable Rules Automatically*.

  - **Reset to Default** resets the page to a default state by removing any persisted settings.

### 3.6.2. Report Pages

Use the *Page* dropdown in the header to switch between the report pages.

By default, when opening a report with a single profile result, the [Details Page](index.html#profiler-report-details-page) is shown. When opening a report with multiple results, the [Summary Page](index.html#profiler-report-summary-page) is selected instead. You can change the default report page in the [Profile](index.html#options-profile) options.

#### Summary Page

The *Summary* page shows a table of all collected results in the report, as well as a list of the most important rule outputs (*Prioritized Rules*) which are ordered by the estimated speedup that could potential be obtained by following their guidance. *Prioritized Rules* are shown by default and can be toggled with the \[R\] button on the upper right of the page.



Summary page with Summary Table and Prioritized Rules.

The *Summary Table* gives you a quick comparison overview across all profiled workloads. It contains a number of important, pre-selected metrics which can be customized as explained below. Its columns can be sorted by clicking the column header. You can transpose the table with the *Transpose* button. Aggregate of all results per each counter metric is shown in the table header along with the column name. You can change the aggregated values by selecting the desired results for multiple metrics simultaneously. When selecting any entry by single-click, a list of its *Prioritized Rules* will be shown below the table. Double-click any entry to make the result the currently active one and switch to the [Details Page](index.html#profiler-report-details-page) page to inspect its performance data. By default, kernel demangled names are simplified, renamed and shown in an optimized manner. This behavior can be changed with [Rename Demangled Names](index.html#options-profile) option. If an auto-simplified name is not useful, you can rename it through a configuration file. You can also persist the updated names directly in the report by double-clicking on the name, renaming and saving the report. Use [Rename Kernels Config Path](index.html#options-profile) option to specify the configuration file which should be used while importing renamed kernels or exporting demangled names with mappings to rename them. To export names to a new file, click *Export* button and use *Rename Kernels Config* option. Hovering over a demangled name cell shows the full name in a tooltip and lets you open the kernel renaming config file from there. See [Kernel Renaming](../NsightComputeCli/index.html#kernel-renaming) for more details on configuration file usage. For results associated with CUDA Graphs, clicking on the result opens the [CUDA Graph Viewer](index.html#tool-window-cuda-graph-viewer) tool window. The viewer displays the instantiated graph structure and automatically highlights the graph node from which the selected kernel was launched, making it easy to understand the execution context within the graph topology.



You can configure the list of metrics included in this table in the [Profile](index.html#options-profile) options dialog. If a metric has multiple instance values, the number of instances is shown after its standard value. A metric with ten instance values could for example look like this: `35.48`` ``{10}`. In the [Profile](index.html#options-profile) options dialog, you can select that all instance values should be shown individually. You can also inspect the instances values of a metric result in the [Metric Details](index.html#tool-window-metric-details) tool window.

In addition to metrics, you can also configure the table to include any of the following properties:

##### Properties

> |  |  |
> |----|----|
> | `property__api_call_id` | ID of the API call associated with this profile result. |
> | `property__block_size` | Block Size. If the result contains multiple launches, this will contain the maximum value for each dimension of the block. |
> | `property__creation_time` | Local collection time. |
> | `property__demangled_name` | Kernel demangled name, potentially renamed. |
> | `property__device_name` | GPU device name. |
> | `property__estimated_speedup` | Maximal relative speedup achievable for this profile result as estimated by the guided analysis rules. |
> | `property__function_name` | Kernel function name. |
> | `property__grid_dimensions` | Grid Dimensions. If the result contains multiple launches, this will contain the maximum value for each dimension of the grid. |
> | `property__grid_offset` | Grid Offset. |
> | `property__grid_size` | Grid Size. If the result contains multiple launches, this will contain the maximum value for each dimension of the grid. |
> | `property__issues_detected` | Number of issues detected by guided analysis rules for this profile result. |
> | `property__kernel_id` | Kernel ID. |
> | `property__mangled_name` | Kernel mangled name. |
> | `property__original_demangled_name` | Original kernel demangled name without any renaming. |
> | `property__process_name` | Process name. |
> | `property__range_name` | Range name. |
> | `property__result_type` | Result Type. This property shows workload type and execution model of the profile result. |
> | `property__runtime_improvement` | Runtime improvement corresponding to the estimated speedup. |
> | `property__series_id` | ID of the profile series. |
> | `property__series_parameters` | Profile series parameters. |
> | `property__thread_id` | CPU thread ID. |
>
> Properties {#id18 .colwidths-auto .table-no-stripes .docutils .align-default}

For [Range Replay](../ProfilingGuide/index.html#range-replay) reports, a smaller set of columns is shown by default, as not all apply to such results.

For the currently selected metric result the *Prioritized Rules* show the most impactful rule results with respect to the estimated potential speedup. Clicking on any of the rule names on the left allows you to easily navigate to the containing section on the details page. With the downward-facing arrow on the right a table with the relevant *key performance indicators* can be toggled. This table contains the metrics which should be tracked when optimizing performance according to the rule guidance.



*Prioritized Rules* with key performance indicators table.

#### Details Page

##### Overview

The *Details* page is the main page for all metric data collected during a kernel launch. The page is split into individual sections. Each section consists of a header table and an optional body that can be expanded. You can expand or collapse the body of each section by clicking on its respective header. The sections are completely user defined and can be changed easily by updating their respective files. For more information on customizing sections, see the [Customization Guide](../CustomizationGuide/index.html#abstract). For a list of sections shipped with NVIDIA Nsight Compute, see [Sections and Rules](../ProfilingGuide/index.html#sections-and-rules).

By default, once a new profile result is collected, all applicable rules are applied. Any rule results will be shown as *Recommendations* on this page. Most rule results will contain an optimization advice along with an estimate of the improvement that could be achieved when successfully implementing this advice. Other rule results will be purely informative or have a warning icon to indicate a problem that occurred during execution (e.g., an optional metric that could not be collected). Results with error icons typically indicate an error while applying the rule.

Estimates of potential improvement are shown below the rule result’s name and exist in two types. *Global estimates* (“Est. Speedup”) are an approximation of the decrease in workload runtime, whereas *local estimates* (“Est. Local Speedup”) are an approximation of the increase in efficiency of the hardware utilization of the particular performance problem the rule addresses.



Rule results often point out performance problems and guide through the analysis process.

If a rule result references another report section, it will appear as a link in the recommendation. Select the link to scroll to the respective section. If the section was not collected in the same profile result, enable it in the [Metric Selection](index.html#tool-window-sections-info) tool window.

You can add or edit comments in each section of the *Details* view by clicking on the comment button (speech bubble). The comment icon will be highlighted in sections that contain a comment. Comments are persisted in the report and are summarized in the [Comments Page](index.html#profiler-report-comments-page).



Use the Comments button to annotate sections.

Besides their header, sections typically have one or more *bodies* with additional charts or tables. Click the triangle *Expander* icon in the top-left corner of each section to show or hide those. If a section has multiple bodies, a dropdown in their top-right corner allows you to switch between them.



Sections with multiple bodies have a dropdown to switch between them.

##### Memory

If enabled, the *Memory Workload Analysis* section contains a Memory chart that visualizes data transfers, cache hit rates, instructions and memory requests. More information on how to use and read this chart can be found in the [Profiling Guide](../ProfilingGuide/index.html#memory-chart).

##### Occupancy

You can open the [Occupancy Calculator](index.html#occupancy-calculator) by clicking on the calculator button in the report header or in the header of the *Occupancy Section*.

##### Range Replay

Note that for [Range Replay](../ProfilingGuide/index.html#range-replay) results some UI elements, analysis rules, metrics or section body items such as charts or tables might not be available, as they only apply to kernel launch-based results. The filters can be checked in the corresponding section files.

##### Rooflines

If enabled, the *GPU Speed Of Light Roofline Chart* section contains a Roofline chart that is particularly helpful for visualizing kernel performance at a glance. (To enable roofline charts in the report, ensure that the section is enabled when profiling.) More information on how to use and read this chart can be found in [Roofline Charts](../ProfilingGuide/index.html#roofline). NVIDIA Nsight Compute ships with several different definitions for roofline charts, including hierarchical rooflines. These additional rooflines are defined in different section files. While not part of the *full* section set, a new section set called *roofline* was added to collect and show all rooflines in one report. The idea of hierarchical rooflines is that they define multiple ceilings that represent the limiters of a hardware hierarchy. For example, a hierarchical roofline focusing on the memory hierarchy could have ceilings for the throughputs of the L1 cache, L2 cache and device memory. If the achieved performance of a kernel is limited by one of the ceilings of a hierarchical roofline, it can indicate that the corresponding unit of the hierarchy is a potential bottleneck.



Sample roofline chart.

The roofline chart can be zoomed and panned for more effective data analysis, using the controls in the table below. When the [Metric Details](index.html#tool-window-metric-details) tool window is open, you can click on an achieved value in the chart to see its metric formula in the tool window.

[TABLE]

Roofline Chart Zoom and Pan Controls {#id20 .table-no-stripes .docutils .align-default}

##### Source

Sections such as *Source Counters* can contain source hot spot tables. These tables indicate the N highest or lowest values of one or more metrics in your kernel source code. Select the location links to navigate directly to this location in the [Source Page](index.html#profiler-report-source-page). Hover the mouse over a value to see which metrics contribute to it.



Hot spot tables point out performance problems in your source.

##### Timelines

When collecting metrics with [PM sampling](../ProfilingGuide/index.html#pm-sampling), they can be viewed in a *timeline*. The timeline shows metrics selected in the respective section file or on the command line with their labels/names and their values over time.



Different metrics may be collected in different passes (replays) of the workload, as only a limited number of them can be sampled in the same pass. Context switch trace is used to filter the collected data to only include samples from the profiled contexts and to align it in the timeline.

You can hover the mouse over a metric row label to see further information on the metrics in the row. Hovering over a sample on the timeline shows the metric values at that timestamp within the current row. With the [Metric Details](index.html#tool-window-metric-details) tool window open, click to select a value on the timeline and show the metric and all its raw timestamps (absolute and relative) correlated values in the tool window.

You can also use the [Metric Details](index.html#tool-window-metric-details) tool window to inspect profiler metrics generated during PM sampling. These provide information about the used sampling intervals, buffer sizes, dropped samples and other properties for each collection pass. A detailed list can be found in the [metrics reference](../ProfilingGuide/index.html#metrics-reference).

The timeline has a context menu for further actions regarding copying, zooming or adjusting the viewed data:

> - Metric rows on the timeline can be scaled (y-axis) based on a theoretical HW peak, or based on the maximum profiled value in the workload. Select the *Scale to Peak*/*Scale to Max Value* options in the context menu to switch between the two modes. Note that HW peak info may not be available for all rows.
>
> - When zoomed out, the bar height may be reduced to represent that multiple samples are aggregated into one bar. The *Show/Hide Max Bars* options in the context menu can be used to enable/disable showing the maximum value for each time range across all samples in that range, even when zoomed out.
>
> - In addition, the *Enable/Disable Context Switch Filter* option can be used to enable or disable the filtering of the timeline data with [context switch](../ProfilingGuide/index.html#pm-sampling) information, if it is available. When the context switch filter is enabled (the default), samples from each pass group are only shown for the active contexts. When the context switch filter is disabled, the raw collected sampling data is shown along with a separate row for each pass group’s context switch trace.
>
> - When the context menu option is not available, the report does not include context switch trace data. In this case, the option *Enable/Disable Workload Alignment* is shown instead if workload execution trace is available. When enabled, it aligns all passes based on their first workload execution timestamp.

The timeline row Workload Execution shows each kernel’s start and end timestamp. When the context switch filter is enabled, kernel execution is only shown for one of the passes for the active contexts. When the context switch filter is disabled, kernel execution is shown for all the passes.

#### Source Page

The *Source* page correlates assembly (SASS) with high-level code such as CUDA-C, Python or PTX. In addition, it displays instruction-correlated metrics to help pinpoint performance problems in your code.



Source Correlation

The page can be switched between different *Views* to focus on a specific source layer or see two layers side-by-side. This includes SASS, PTX and Source (CUDA-C, Fortran, Python, …), as well as their combinations. Which options are available depends on the source information embedded into the executable.

The high-level Source (CUDA-C) view is available if the application was built with the `-lineinfo` or `--generate-line-info` nvcc flag to correlate SASS and source. When using separate linking at the ELF level, there is no PTX available in the ELF that would correspond to the final SASS. As such, NVIDIA Nsight Compute does not show any PTX even though it would be available statically in the executable and could be shown with `cuobjdump`` ``-all`` ``-lptx`. However, this is a pre-linked version of the PTX and cannot be reliably used for correlation.

##### Navigation

The *View* dropdown can be used to select different code (correlation) options: SASS, PTX and Source (CUDA-C, Fortran, Python, …).

In side-by-side views, when selecting a line in the left-hand- or right-hand-side, any correlated lines in the opposite view are highlighted. However, when the [Show Single File For Multi-File Sources](index.html#options-profile) option is set to *Yes*, the target file or source object must already be selected in the respective view for those correlated lines to be shown.

The locations of correlated lines are displayed on the minimap. Clicking on the minimap will scroll to the corresponding source line. The source correlation navigation controls allow you to move to the previous or next block of correlated lines.



Correlation Navigation

In side-by-side views, the view which has blue border is called *focused view*. Focused view can be changed by clicking anywhere in the other view. Control groups which are bordered with blue color will work for a focused view only.

The *Source* dropdown allows you to switch between the files or functions that provide the content in the view. When a different source entry is selected, the view scrolls to the start of this file or function.

If a view contains multiple source files or functions, \[+\] and \[-\] buttons are shown. These can be used to expand or collapse the view, thereby showing or hiding the file or function content except for its header. These will work for both the views if that group is in the linked state otherwise in an unlinked state it will work for a focused view. If collapsed, all [metrics](index.html#profiler-report-source-page-metrics) are shown aggregated to provide a quick overview.



Collapsed Source View

The SASS instruction disassembly includes a symbol label column. Symbol labels are displayed alongside instruction addresses, matching the output style of the `nvdisasm` tool. These labels are interactive hyperlinks and clicking a symbol label will immediately navigate to the corresponding address line within the view. Labels are shown for branch targets and function calls, making it easy to follow control flow and symbol references directly from the disassembly. This feature allows you to quickly explore code flow and symbol references.



Symbol Labels

You can use the search functionality to find information within your data efficiently. This feature allows users to search within a specific column by selecting it through a click anywhere in the column. If no column is selected, the search defaults to the *Source* column. The functionality supports advanced search expressions, such as regular expressions (regex) and value comparisons (e.g., “\>= 5”). The search control can be opened using the search button on the common control toolbar or by pressing the Ctrl+F keys. It can be closed using the cross button or by pressing the Esc key.



Search Control

The SASS view is filtered to only show functions that were executed in the launch. You can toggle the [Show Only Executed Functions](index.html#options-profile) option to change this, but performance of this page may be negatively affected for large binaries. It is possible that some SASS instructions are shown as *N/A*. Those instructions are not currently exposed publicly.

In side-by-side views, the *Navigate By* dropdowns are linked with each other by default, thereby changing column names from one dropdown will change it in the other view only if column is available. These dropdowns can be unlinked with the link-unlink button provided just before it.



Linked state of “Navigate By” Dropdowns

Only filenames are shown in the view, together with a *File Not Found* error, if the source files cannot be found in their original location. This can occur, for example, if the report was moved to a different system. Select a filename and click the *Resolve* button above to specify where this source can be found on the local filesystem. However, the view always shows the source files if the [import source](index.html#connection-activity-interactive) option was selected during profiling, and the files were available at that time. If a file is found in its original or any source lookup location, but its attributes don’t match, a *File Mismatch* error is shown. See the [Source Lookup](index.html#options-source-lookup) options for changing file lookup behavior.



Resolve Source

If the report was collected using remote profiling, and automatic resolution of remote files is enabled in the [Profile](index.html#options-profile) options, NVIDIA Nsight Compute will attempt to load the source from the remote target. If the connection credentials are not yet available in the current NVIDIA Nsight Compute instance, they are prompted in a dialog. Loading from a remote target is currently only available for Linux x86_64 targets and Linux and Windows hosts.

##### Metrics

###### Metrics Correlation

The page is most useful when inspecting performance information and metrics correlated with your code. Metrics are shown in columns, which can be enabled or disabled using the *Column Chooser* accessible using the column header right click menu.



Column Chooser

To not move out of view when scrolling horizontally, columns can be fixed. By default, the *Source* column is fixed to the left, enabling easy inspection of all metrics correlated to a source line. To change fixing of columns, right click the column header and select *Freeze* or *Unfreeze*, respectively.



Column Freezing/Unfreezing

The heatmap on the right-hand side of each view can be used to quickly identify locations with high metric values of the currently selected metric in the dropdown. The heatmap uses a black-body radiation color scale where black denotes the lowest mapped value and white the highest, respectively. The current scale is shown when clicking and holding the heatmap with the right mouse button.



Heatmap Color Scale

By default, applicable metrics are shown as percentage values relative to their sum across the launch. A bar is filling from left to right to indicate the value at a specific source location relative to this metric’s maximum within the launch. The \[%\] and \[+-\] buttons can be used to switch the display from relative to absolute and from abbreviated absolute to full-precision absolute, respectively. For relative values and bars, the \[circle/pie\] button can be used to switch the display between relative to global (launch) and relative to local (function/file) scope. This button is disabled when the view is collapsed, as percentages are always relative to the global launch scope in this case.



Relative and Absolute Metric Values.

###### Pre-Defined Source Metrics

- **Metric Pipelines**

  The [metric pipelines](../ProfilingGuide/index.html#pipelines) this instruction (can) execute in. Some instructions can execute in one of multiple pipelines, and the decision is dynamic at runtime. Some instructions execute in both of two pipelines, and the association is static. Other instructions execute in only a single pipeline.

- **Scoreboard Dependencies**

  > See [Instructions & Scoreboards Table](../NsightCompute/index.html#profiler-report-source-page-additional-tables) for information on the scoreboard dependencies.

- **Live Registers**

  Number of registers that need to be kept valid by the compiler. A high value indicates that many registers are required at this code location, potentially increasing the register pressure and the maximum number of register required by the kernel.

  The total number of registers reported as `launch__registers_per_thread` may be significantly higher than the maximum live registers. The compiler may need to allocate specific registers that can creates holes in the allocation, thereby affecting `launch__registers_per_thread`, even if the maximum live registers is smaller. This may happen due to ABI restrictions, or restrictions enforced by particular hardware instructions. The compiler may not have a complete picture of which registers may be used in either callee or caller and has to obey ABI conventions, thereby allocating different registers even if some register could have theoretically been re-used.

- **Attributed Stalls**

  Number of warp stall samples of type long/short scoreboard not issued, attributed to this scoreboard producer location.

  Use this to identify source locations that cause the most stalls. The stalls are attributed to the producer location, which is the location of the line that produced the scoreboard. This may not be the same as the location of the instruction that is stalled, as it may be waiting on a scoreboard produced by another line.

- **Instruction Mix/Category**

  Instruction Mix (high-level source) or Instruction Category (SASS) show the breakdown of, or the instruction category, respectively, for each line.

- **Warp Stall Sampling (All Samples)**[1](#fn1)

  The number of samples from the [Statistical Sampler](../ProfilingGuide/index.html#statistical-sampler) at this program location.

- **Warp Stall Sampling (Not-issued Samples)**[2](#fn2)

  The number of samples from the [Statistical Sampler](../ProfilingGuide/index.html#statistical-sampler) at this program location on cycles the warp scheduler issued no instructions. Note that *(Not Issued)* samples may be taken on a different profiling pass than *(All)* samples mentioned above, so their values do not strictly correlate.

  This metric is only available on devices with compute capability 7.0 or higher.

- **Instructions Executed**

  Number of times the source (instruction) was executed per individual warp, independent of the number of participating threads within each warp.

- **Thread Instructions Executed**

  Number of times the source (instruction) was executed by any thread, regardless of predicate presence or evaluation.

- **Predicated-On Thread Instructions Executed**

  Number of times the source (instruction) was executed by any active, predicated-on thread. For instructions that are executed unconditionally (i.e. without predicate), this is the number of active threads in the warp, multiplied with the respective *Instructions Executed* value.

- **Avg. Threads Executed**

  Average number of thread-level executed instructions per warp, regardless of their predicate.

- **Avg. Predicated-On Threads Executed**

  Average number of predicated-on thread-level executed instructions per warp.

- **Divergent Branches**

  Number of divergent branch targets, including fallthrough. Incremented only when there are two or more active threads with divergent targets. Divergent branches can lead to warp stalls due to resolving the branch or instruction cache misses.

- **Information on Memory Operations**

  |  |  |  |
  |----|----|----|
  | **Label** | **Name** | **Description** |
  | Address Space | memory_type | The accessed address space (global/local/shared). |
  | Access Operation | memory_access_type | The type of memory access (e.g. load or store). |
  | Access Size | memory_access_size_type | The size of the memory access, in bits. |
  | L1 Tag Requests Global | memory_l1_tag_requests_global | Number of L1 tag requests generated by global memory instructions. |
  | L1 Conflicts Shared N-Way | derived\_\_memory_l1_conflicts_shared_nway | Average N-way conflict in L1 per shared memory instruction. A 1-way access has no conflicts and resolves in a single pass. Note: This is a derived metric which can not be collected directly. |
  | L1 Wavefronts Shared Excessive | derived\_\_memory_l1_wavefronts_shared_excessive | Excessive number of wavefronts in L1 from shared memory instructions, because not all not predicated-off threads performed the operation. Note: This is a derived metric which can not be collected directly. |
  | L1 Wavefronts Shared | memory_l1_wavefronts_shared | Number of wavefronts in L1 from shared memory instructions. |
  | L1 Wavefronts Shared Ideal | memory_l1_wavefronts_shared_ideal | Ideal number of wavefronts in L1 from shared memory instructions, assuming each not predicated-off thread performed the operation. |
  | L2 Theoretical Sectors Global Excessive | derived\_\_memory_l2_theoretical_sectors_global_excessive | Excessive theoretical number of sectors requested in L2 from global memory instructions, because not all not predicated-off threads performed the operation. Note: This is a derived metric which can not be collected directly. |
  | L2 Theoretical Sectors Global | memory_l2_theoretical_sectors_global | Theoretical number of sectors requested in L2 from global memory instructions. |
  | L2 Theoretical Sectors Global Ideal | memory_l2_theoretical_sectors_global_ideal | Ideal number of sectors requested in L2 from global memory instructions, assuming each not predicated-off thread performed the operation. |
  | L2 Theoretical Sectors Local | memory_l2_theoretical_sectors_local | Theoretical number of sectors requested in L2 from local memory instructions. |

  All *L1/L2 Sectors/Wavefronts/Requests* metrics give the number of achieved (actually required), ideal, and excessive (achieved - ideal) sectors/wavefronts/requests. *Ideal* metrics indicate the number that would needed, given each not predicated-off thread performed the operation of given width. *Excessive* metrics indicate the required surplus over the ideal case. Reducing divergence between threads can reduce the excess amount and result in less work for the respective HW units.

Several of the above metrics on memory operations were renamed in version 2021.2 as follows:

|                                |                                            |
|--------------------------------|--------------------------------------------|
| **Old name**                   | **New name**                               |
| memory_l2_sectors_global       | memory_l2_theoretical_sectors_global       |
| memory_l2_sectors_global_ideal | memory_l2_theoretical_sectors_global_ideal |
| memory_l2_sectors_local        | memory_l2_theoretical_sectors_local        |
| memory_l1_sectors_global       | memory_l1_tag_requests_global              |
| memory_l1_sectors_shared       | memory_l1_wavefronts_shared                |
| memory_l1_sectors_shared_ideal | memory_l1_wavefronts_shared_ideal          |

- **L2 Explicit Evict Policy Metrics**

  Starting with the NVIDIA Ampere architecture the eviction policy of the L2 cache can be tuned to match the kernel’s access pattern. The eviction policy can be either set implicitly for a memory window (for more details see [CUaccessProperty](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaAccessPolicyWindow.html)) or set explicitly per executed memory instruction. If set explicitly, the desired eviction behavior for the cases of an L2 cache hit or miss are passed as input to the instruction. For more details refer to CUDA’s [Cache Eviction Priority Hints](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-eviction-priority-hints).

  |  |  |  |
  |----|----|----|
  | **Label** | **Name** | **Description** |
  | L2 Explicit Evict Policies | smsp\_\_inst_executed_memdesc_explicit_evict_type | Comma separated list of configured explicit eviction policies. As the policies can be set dynamically at runtime, this list includes all policies that were part of any executed instruction. |
  | L2 Explicit Hit Policy Evict First | smsp\_\_inst_executed_memdesc_explicit_hitprop_evict_first | Number of times a memory instruction was executed by any warp which had the `evict_first` policy set in case the access leads to a cache hit in L2. Data cached with this policy will be first in the eviction priority order and will likely be evicted when cache eviction is required. This policy is suitable for streaming data. |
  | L2 Explicit Hit Policy Evict Last | smsp\_\_inst_executed_memdesc_explicit_hitprop_evict_last | Number of times a memory instruction was executed by any warp which had the `evict_last` policy set in case the access leads to a cache hit in L2. Data cached with this policy will be last in the eviction priority order and will likely be evicted only after other data with `evict_normal` or `evict_first` eviction policy is already evicted. This policy is suitable for data that should remain persistent in cache. |
  | L2 Explicit Hit Policy Evict Normal | smsp\_\_inst_executed_memdesc_explicit_hitprop_evict_normal | Number of times a memory instruction was executed by any warp which had the `evict_normal` (default) policy set in case the access leads to a cache hit in L2. |
  | L2 Explicit Hit Policy Evict Normal Demote | smsp\_\_inst_executed_memdesc_explicit_hitprop_evict_normal_demote | Number of times a memory instruction was executed by any warp which had the `evict_normal_demote` policy set in case the access leads to a cache hit in L2. |
  | L2 Explicit Miss Policy Evict First | smsp\_\_inst_executed_memdesc_explicit_missprop_evict_first | Number of times a memory instruction was executed by any warp which had the `evict_first` policy set in case the access leads to a cache miss in L2. Data cached with this policy will be first in the eviction priority order and will likely be evicted cache eviction is required. This policy is suitable for streaming data. |
  | L2 Explicit Miss Policy Evict Normal | smsp\_\_inst_executed_memdesc_explicit_missprop_evict_normal | Number of times a memory instruction was executed by any warp which had the `evict_normal` (default) policy set in case the access leads to a cache miss in L2. |

- **Individual Warp Stall Sampling Metrics**

  All *stall\_\** metrics show the information combined in *Warp Stall Sampling* individually. See [Statistical Sampler](../ProfilingGuide/index.html#statistical-sampler) for their descriptions.

- See the [Customization Guide](../CustomizationGuide/index.html#abstract) on how to add additional metrics for this view and the [Metrics Reference](../ProfilingGuide/index.html#metrics-reference) for further information on available metrics.

###### Register Dependencies

Dependencies between registers are displayed in the SASS view. When a register is read, all the potential addresses where it could have been written are found. The links between these lines are drawn in the view. All dependencies for registers, predicates, uniform registers and uniform predicates are shown in their respective columns.

[](../_images/profiler-report-pages-source-register-dependencies.png)

Register Dependencies

The picture above shows some dependencies for a simple CUDA kernel. On the first row, which is line 9 of the SASS code, we can see *writes* on registers R2 and R3, represented by *filled triangles pointing to the left*. These registers are then read on lines 17, 20 and 23, and this is represented by *regular triangles pointing to the right*. There are also some lines where both types of triangles are on the same line, which means that a read and a write occured for the same register.

The lines are colored with a gradient that helps visualizing how long the dependencies are.

Dependencies across source files and functions are not tracked.

The Register Dependencies Tracking feature is enabled by default, but can be disabled completely in *Tools \> Options \> Profile \> Report Source Page \> Enable Register Dependencies*.

[1](#id6)  
This metric was previously called Sampling Data (All).

[2](#id7)  
This metric was previously called Sampling Data (Not Issued).

##### Profiles

The icon next to the *View* dropdown can be used to manage *Source View Profiles*.



This button opens a dialog that shows you the list of saved source view profiles. Such profiles can be created using the `+` button in the dialog. Profiles let you store the column properties of all views in the report to a file. Such properties include column visibility, freeze state, width, order and the selected navigation metric. Double-click on a saved profile to apply it to any opened report. This updates the column properties mentioned above from the selected profile in all views.



Profiles are useful for configuring views to your preferences, or for a certain use case. Start by choosing metric columns from the *Column Chooser*. Next, configure other properties like freezing column, changing width or order and setting a heatmap metric in the *Navigation* dropdown before creating the profile. Once a profile is created, you can always use this profile on any opened report to hide all non-required columns or to restore your configured properties. Simply select the profile from the source view profiles dialog and apply it. You can also make the profile default by clicking on the *star* button to automatically apply it while opening a report.

Note that the column properties are stored separately for each *View* in the profile and when applied, only those views will be updated which are present in the selected profile. You will not see the metric columns that are not available in your report even if those were configured to be visible in the source profile you have applied.

##### Additional Tables

###### Instructions & Dependencies Table

If **Scoreboard Dependencies** is selected, this table shows the following information for the selected line:

- **Self Instruction Mix**: The breakdown of all instruction categories. Use this to understand the composition of the selection, and which instruction categories have the most (scoreboard) warp stalls.

- **Input Scoreboard Dependencies**: The scoreboard-driven data dependencies of the selection. Data dependencies are predecessor operations that need to complete before code in the selection can proceed. Use this to understand on which instructions the selection is waiting (stalled) on.

  The *Attributed Stalls* column shows the number of warp stalls from the selection attributed to this dependency. As a result, dependencies causing more stalls in the selection are sorted higher in the table. This data is also available in the source views to locate lines causing the most stalls.

- **Output Scoreboard Dependencies**: The consumers of scoreboards produced by the selection. A scoreboard producer is an operation producing a scoreboard a consumer is waiting on. Use this to understand who is waiting (stalled) on the selection. Note that these lines may wait for other instructions than the current selection, too.

Scoreboards are used to synchronize data dependencies, state updates and ensure memory ordering. This includes true data dependencies (read-after-write) and anti-dependencies (write-after-read).

Relative percentage values in these tables are calculated based on the total value of the respective data in the entire view, not just in the selection.

The *Dependencies* tables show in their *Scoreboards* columns which of the GPU’s six scoreboards are responsible for the dependency. Use the *Scoreboard Dependencies* column in the SASS view to see individual scoreboard dependencies between assembly instructions. Lines at the same offset in this column are for the same scoreboard.

The *Scoreboard Stalls* column uses different colors to distinguish between long and short scoreboard warp stalls.



Scoreboard Dependencies Table

If **Register Dependencies** is selected, this table shows the following information for the selected line:

- **Self Instruction Mix**: The breakdown of all instruction categories. Use this to understand the composition of the selection.

- **Input Register Dependencies**: The general purpose register-driven data dependencies of the selection. Data dependencies are predecessor operations that need to complete before code in the selection can proceed. Use this to understand which instructions produce/write registers the selection is waiting to consume/read.

  The *Attributed Live Registers* column shows the number of live general-purpose registers attributed to instructions of this category. This is the maximum number of live registers across all consumers of dependencies from this producer. This data is also available in the source views to locate lines causing the most live registers.

  The *Output Registers* column shows the number of register dependencies produced by instructions of this category. This is the sum of all general-purpose registers written by this producer. This data is also available in the source views to locate lines producing the most dependencies.

- **Output Register Dependencies**: This table lists all consumers/readers of general-purpose registers produced/written by the selected line. A register producer is an operation writing a register a consumer is waiting on for reading. Use this to understand which instructions are waiting on the selection to produce/write registers. Note that these lines may wait for other instructions than the current selection, too.



Register Dependencies Table

For both scoreboard and register views, the dependencies are grouped by location and instruction category. If the location does not match the selected line, it is shown as a clickable link. Clicking on the link will navigate to the corresponding line. If the selection is in a high-level (e.g., CUDA, Python) view, the locations are mapped to the high-level language if possible. If not possible due to missing correlation information, or if the selection is in the SASS view, the locations are shown in the low-level language.

###### Inline Functions Table

This table shows all call sites where the selected function from the *Source* view is inlined.

In the *Source* view, correlated metric values of every inline function source line are an aggregation of metric values of related SASS lines from all call sites. The table provides such SASS lines info for each call site individually to help in identifying which call site contributes how much to the overall metric values.



Inline Functions Table

###### Source Markers Table

The code in the different *Views* can also contain warnings, errors or just notifications that are displayed as *Source Markers* in the left header, as shown below. These can be generated from multiple sources, but as of now only NvRules are supported.



Source Markers

This table shows all lines containing *Source Markers* in the open View. The user can click on the line number to navigate to the corresponding line in the view.



Source Markers Table

###### Statistics Table

This table allows the user to select multiple lines in the source code above and calculate a variety of statistics. They can be used to check if anything seems unusual, like an unexpected gap between minimum and maximum.

Cells will be empty if the metric has either not been collected or its statistical value is not meaningful in the context of the calculation. For example, a memory access or any other non-numeric metric can’t have a minimum or maximum since there is no defined order.



Statistics Table

##### Limitations

Graph Profiling

When profiling complete CUDA graphs, instruction-level source metrics are not available.

#### Context Page

The *CPU Call Stack* section of this report page shows the CPU call stack(s) for the executing CPU thread at the time the kernel was launched. For this information to show up in the profiler report, the option to collect CPU call stacks had to be enabled in the [Connection Dialog](index.html#connection-dialog) or using the corresponding NVIDIA Nsight Compute CLI command line parameter.



NVIDIA Nsight Compute supports to collect *native* CPU call stacks as well as call stacks for *Python* applications. Either or both types can be selected in the *Activity* menu of the [Connection Dialog](index.html#connection-dialog) (via the “CPU Call Stack Types” option), or using the NVIDIA Nsight Compute [CLI command line parameter](../NsightComputeCli/index.html#command-line-options-launch) –call-stack-type. In case both types are enabled, a dropdown menu will appear to select the desired call stack type.



Note that Python call stack collection requires CPython version 3.9 or later.

The *NVTX State* section of this report page shows the NVTX context when the kernel was launched. All thread-specific information is with respect to the thread of the kernel’s launch API call. Note that NVTX information is only collected if the profiler is started with NVTX support enabled, either in the [Connection Dialog](index.html#connection-dialog) or using the NVIDIA Nsight Compute CLI command line parameter.



This page has been renamed from “Call Stack / NVTX Page”.

#### Comments Page

The *Comments* page aggregates all section comments in a single view and allows the user to edit those comments on any launch instance or section, as well as on the overall report. Comments are persisted with the report. If a section comment is added, the comment icon of the respective section in the [Details Page](index.html#profiler-report-details-page) will be highlighted.

#### Raw Page

The *Raw* page shows a list of all collected metrics with their units per profiled kernel launch. It can be exported, for example, to CSV format for further analysis. The page features a filter edit to quickly find specific metrics. You can transpose the table of kernels and metrics by using the *Transpose* button.

If a metric has multiple instance values, the number of instances is shown after the standard value. This metric for example has ten instance values: `35.48`` ``{10}`. You can select in the [Profile](index.html#options-profile) options dialog that all instance values should be shown individually or inspect the metric result in the [Metric Details](index.html#tool-window-metric-details) tool window.

#### Session Page

This *Session* page contains basic information about the report and the machine, as well as device attributes of all devices for which launches were profiled. When switching between launch instances, the respective device attributes are highlighted.

### 3.6.3. Metrics and Units

Numeric metric values are shown in various places in the report, including the header and tables and charts on most pages. NVIDIA Nsight Compute supports various ways to display those metrics and their values.

When available and applicable to the UI component, metrics are shown along with their unit. This is to make it apparent if a metric represents cycles, threads, bytes/s, and so on. The unit will normally be shown in rectangular brackets, e.g. `Metric`` ``Name`` ``[bytes]`` ``128`.

By default, units are scaled automatically so that metric values are shown with a reasonable order of magnitude. Units are scaled using their SI-factors, i.e. byte-based units are scaled using a factor of 1000 and the prefixes K, M, G, etc. Time-based units are also scaled using a factor of 1000, with the prefixes n, u and m. This scaling can be disabled in the [Profile](index.html#options-profile) options.

Metrics which could not be collected are shown as `n/a` and assigned a warning icon. If the metric floating point value is out of the regular range (i.e. `nan` (Not a number) or `inf` (infinite)), they are also assigned a warning icon. The exception are metrics for which these values are expected and which are allow-listed internally.

### 3.6.4. Filtered Profiler Report

A filtered profiler report is a subset of the full profiler report. You can save filtered or selected results from the full profiler report to a new file. To save the filtered results, you can use the File \> Save Filtered Results As menu option, which will save the [Filtered Results](../NsightCompute/index.html#profiler-report-header-filter-dialog) to a new file. Alternatively, you can select one or more results on the [Summary Page](../NsightCompute/index.html#summary-page) or [Raw Page](../NsightCompute/index.html#raw-page) and save them to a new file using the File \> Save Selected Results As menu option or the right-click Save Result(s) menu option.

## 3.7. Baselines

NVIDIA Nsight Compute supports diffing collected results across one or multiple reports using Baselines. Each result in any report can be promoted to a baseline. This causes metric values from all results in all reports to show the difference to the baseline. If multiple baselines are selected simultaneously, metric values are compared to the average across all current baselines. Baselines are not stored with a report and are only available as long as the same NVIDIA Nsight Compute instance is open, unless they are saved to a `ncu-bln` file from the [Baselines tool window](index.html#tool-window-baselines).



Profiler report with one baseline

Select *Add Baseline* to promote the current result in focus to become a baseline. If a baseline is set, most metrics on the [Details Page](index.html#profiler-report-details-page), [Raw Page](index.html#profiler-report-raw-page) and [Summary Page](index.html#profiler-report-summary-page) show two values: the current value of the result in focus, and the corresponding value of the baseline or the percentage of change from the corresponding baseline value. (Note that an infinite percentage gain, *inf%*, may be displayed when the baseline value for the metric is zero, while the focus value is not.)

If multiple baselines are selected, each metric will show the following notation:

``` text
<focus value> (<difference to baselines average [%]>, z=<standard score>@<number of values>)
```

The standard score is the difference between the current value and the average across all baselines, normalized by the standard deviation. If the number of metric values contributing to the standard score equals the number of results (current and all baselines), the @\<number of values\> notation is omitted.



Profiler report with multiple baselines

Baseline added for the current result in focus is always shown on the top. However, the actual order of added baselines is shown in the [Baselines tool window](index.html#tool-window-baselines).



Baselines tool window with mutliple baselines

Double-clicking on a baseline name allows the user to edit the displayed name. Edits are committed by pressing `Enter/Return` or upon loss of focus, and abandoned by pressing `Esc`. Hovering over the baseline color icon allows the user to remove this specific baseline from the list.

Use the *Clear Baselines* entry from the dropdown button, the [Profile](index.html#options-profile) menu, or the corresponding toolbar button to remove all baselines.

Baseline changes can also be made in the [Baselines tool window](index.html#tool-window-baselines).

## 3.8. Standalone Source Viewer

NVIDIA Nsight Compute includes a standalone source viewer for *cubin* files. This view is identical to the [Source Page](index.html#profiler-report-source-page), except that it won’t include any performance metrics.

Cubin files can be opened from the *File* \> *Open* main menu command. The SM Selection dialog will be shown before opening the standalone source view. If available, the SM version present in the file name is pre-selected. For example, if your file name is `mergeSort.sm_80.cubin` then SM 8.0 will be pre-selected in the dialog. Choose the appropriate SM version from the drop down menu if it’s not included in the file name.



SM Selection Dialog

Click Ok button to open [Standalone Source Viewer](index.html#cubin-viewer).



Standalone Source Viewer

## 3.9. Source Comparison

Source comparison provides a way to see the source files of two profile results side by side. It enables to quickly identify source differences and understand changes in metric values.

To compare two results side by side add one result as a baseline, navigate to the other result, and then click the *Source Comparison* button located in the report header.

For example, if you want to compare kernel XYZ from report R1 with kernel XYZ from report R2, first open report R1, add the profile result for kernel XYZ as baseline, open report R2, choose kernel XYZ, and then click the Source Comparison button.

Source comparison will be shown only with first added baseline result.



Source Comparison Button



Source Comparison

Currently only high-level Source (CUDA-C) view and SASS view are supported for comparison. The source difference heatmap is located on the very left side. Each of its partitions represents a side of the source diff. Clicking on the heatmap will scroll to the respective source line.

Navigation to the previous or next difference is supported using the navigation buttons or the keyboard shortcuts *Ctrl + 1* and *Ctrl + 2*.



Source Comparison Navigation Buttons

On the SASS view, the *Diff By* drop down menu allows you to choose the diff basis based on either *Opcode* or *Full Instruction*. For the latter, all instruction modifiers and arguments are considered for the comparison in addition to the opcode.



Source Comparison Diff By Menu

## 3.10. Report Merge Tool

NVIDIA Nsight Compute provides a *Report Merge Tool* utility that enables users to combine multiple Nsight Compute reports with the `.ncu-rep` extension into a single file. It is particularly useful for multi-GPU systems and scenarios when comparing and analyzing several reports individually becomes impractical. The report created with ReportMergeTool is fully compatible with Nsight Compute, and can be used for visualization and analysis in the UI and on the command line.

There are 2 options for using the MergeTool.

1.  **Launching from the UI menu**

    Select the Merge Tool item from Tools in the [Main Menu](index.html#main-menu).

2.  **Launching from the Command Line**

    Call the binary in `$NCU_INSTALL_PATH/extras/ReportUtils/ReportMergeTool`.

    It accepts a directory as an input, and it will merge all the reports under that directory. Other features follow exactly the same logic as the UI.

### 3.10.1. File Selection Tab

This tab handles the selection of reports and allows to set the output name and/or path for the resulting merged report. There are 2 ways of selecting report files to merge.

1.  **Selecting from system files**

    This option shows a hierarchical view of `.ncu-rep` files in the system, and the containing folders. Individual reports can be selected together with entire folders. The *Report Directory* input changes the root directory of the hierarchical view.

    
    Selection from System Files

2.  **Selecting from currently open reports**

    When some reports are already open in Nsight Compute, MergeTool will suggest to merge them first. To change select from the system files, set *Select from* combo box item to *System Files*, or use the *Report Directory* selector.

### 3.10.2. Metric Filters Tab

This tab guides the selection and merging process of individual metrics. Selecting the *Merge Operation* defines which metrics are merged and kept across all report results. *Aggregation Operation* then tells how these metrics are merged, and how the resulting value is calculated from multiple values. For any other filtering operation metric names can be provided in *Results* text box, or *Metric Sets/Sections* can be selected.



Metric Filter Tab

Metric Merge Operations

Metric Merge Operations dictate how the metric set for matching results is maintained in the merged report.

- **Union** keeps all the metrics that appear at least once across the matching results in all reports. There can only be one metric set per result, metrics that don’t exist in some reports will appear empty in those cases.

- **Intersection** keeps only the metrics that exist in all matching results across the reports. This option guarantees that no metrics will have missing values in the output. Note that if reports contain no matching results, they cannot be merged, which may result in some metrics being empty in the final output.

Aggregation Operations

Whenever two or more report results are merged, the metric values are aggregated. Aggregation operations tell the MergeTool how the metrics are combined together. As a result, the merged report will contain one of these four options as a metric value.

- **Average** of the combined metric values

- **Sum**

- **Minimum**

- **Maximum**

Metric Selection: Metric List, Sets, Sections

Apart from Merge Operations, the metric list for the output report can be manipulated using filtering. The simplest way to set the metric list is to enter the metrics as a comma-separated list in the *Metrics* text box.



Metric Sets/Sections

Additionally, metric sets and sections can be used, by loading a view similar to [Metric Selection](index.html#metric-selection). Selecting the sets and sections modifies the appearance of the reports by changing the sections that are displayed on the *Details* page. The metrics of the reports are not affected. To remove unwanted sections, select the *Remove Unselected Sections* checkbox.

### 3.10.3. Result Filters Tab

This tab guides the selection and merging process of individual results. It allows to select the [Merge Operation](index.html#mergetool-results-merge-op) to define which results are merged and kept.

For any other filtering operations result names can be provided in the *Results* text box as a comma-separated list. If this list is set, only the provided results will appear in the output.



Result Filter Tab

#### Result Merge Operations

Result merge operations, similarly to the metric merge operations, tell the MergeTool which results to keep and which results to discard. Additionally, they can manage the merge process by matching the results from the merged reports. The matched results are the ones that will be aggregated together into a single result. The four available options provide control over which results are merged together and how to handle unmatched results.

- **Union** merges only matching results from every report, keeps unmatched results as is. When multiple iterations of the same kernel appear in a report, they are kept separate and are only merged with the corresponding iteration of this kernel in other reports.

  *For example:* Merging Report1 with results name {kernel_A, kernel_B} and Report2 with {kernel_A, kernel_C} results in {kernel_A^(merged) , kernel_B, kernel_C}. i.e.,

  \\\\A, B\\ + \\A, C\\ = \\A^{merged}, B, C\\\\

  *Note:* if multiple results share the same name, only matching ones are merged. i.e.,

  \\\\A\_{1}, A\_{2}, A\_{3}\\ + \\A\_{1}, A\_{2}\\ = \\A\_{1}^{merged}, A\_{2}^{merged}, A\_{3}\\\\

- **Intersection** merges matching results from every report in the same way as *Union* does, but discards all the unmatched results.

  \\\\A, B\\ + \\A, C\\ = \\A^{merged}\\\\

- **Collapse** merges all results sharing the same name. When multiple iterations of the same kernel appear in a report, merges them together.

  \\\\A\_{1}, A\_{2}, A\_{3}\\ + \\A\_{1}, A\_{2}\\ = \\A^{merged}\\\\

- **Concat** keeps all the results in their original form and saves them in a single file, no metric aggregation is done.

  \\\\A, B\\ + \\A, C\\ = \\A\_{1}, B, A\_{2}, C\\\\

#### Merged Report

To be able to track how a merged report was created, each merged report contains a comment with the list of reports that were merged, the merge operation, the aggregation operation and the sections that were selected.



Comment with merge details in the merged report

## 3.11. Clustering Window

The **Clustering Window** helps you analyze and compare multiple profiling reports by grouping similar reports together. This makes it easier to identify performance patterns and find relationships between different profiling sessions.



Clustering Window

### 3.11.1. How to Use Clustering

1.  **Open Clustering Window**: From the main menu, select *Tools* \> *Clustering Window*. The window can be docked or undocked like other tool windows.

2.  **Select Reports**: Click *Select Reports* and choose multiple `.ncu-rep` files from the file dialog.

3.  **Configure Parameters** (Optional):

    - *Minimum Cluster Size*: How many reports are needed to form a cluster (default: 3, minimum allowed: 2)

    - *Maximum Cluster Size*: Maximum reports per cluster (0 = no limit)

4.  **Run Analysis**: Click *Cluster Reports* to analyze the selected reports and generate clusters.

### 3.11.2. Clustering Tree



Clustering Tree

The clustering tree shows how reports are grouped together hierarchically:

- **Reports and Clusters**: Individual reports are shown as leaf nodes in the tree with unique IDs, and clusters are shown as nodes with a **+** sign.

- **Selecting Clusters**: Click the **+** nodes to select a cluster for merging. Selected clusters will be shown in green, and the corresponding entries in the similarity matrix below will be highlighted with a dotted border.

- **Highlighting**: When you hover over any cluster or report in the tree, the corresponding entries are highlighted in the tree and the similarity matrix below.

The tree structure helps you understand:

- Which reports are most similar to each other (grouped in the same cluster)

- The hierarchical relationships between different groups of reports

- How the clustering algorithm has organized your data

### 3.11.3. Similarity Matrix



Similarity Matrix

The similarity matrix displays a grid showing how similar each pair of reports is between each other. This matrix is used to generate the clustering tree - reports with higher similarity scores are grouped closer together in the tree structure.

- **Rows and Columns**: Each row displays a report name along with its ID, while columns show report IDs that are used in the Clustering Tree.

- **Similarity Scores**: Each cell contains a similarity value calculated based on the metric values from the reports.

- **Color Coding**:

  - Darker colors indicate higher similarity between reports

  - Lighter colors indicate lower similarity

  - The highlighted entries in the similarity matrix are the ones that are being hovered over in the clustering tree.

- **Similarity Score Values**: Hover over any cell to see the exact similarity score between those two reports.

Control Buttons

- *Select Reports* opens a file dialog to choose the `.ncu-rep` files you want to analyze.

- *Run Clustering* starts the clustering analysis using the selected reports and current parameters.

- *Reset to Default* restores clustering parameters to their default values.

- *Clear Clusters* clears the current cluster selection.

- *Suggest Clusters* automatically recommends optimal clustering parameters based on your data characteristics. The suggestion will cover most of the reports, but outliers may be excluded.

- *Merge Clusters* creates a merged report from the selected clusters. The merged reports can be processed using the [Report Merge Tool](index.html#report-merge-tool.html) for further analysis and customization.

## 3.12. Occupancy Calculator

NVIDIA Nsight Compute provides an *Occupancy Calculator* that allows you to compute the multiprocessor occupancy of a GPU for a given CUDA kernel. It replaces the previously provided CUDA Occupancy Calculator spreadsheet.

The Occupancy Calculator can be opened directly from a profile report or as a new activity. The occupancy calculator data can be saved to a file using *File \> Save*. By default, the file uses the `.ncu-occ` extension. The occupancy calculator file can be opened using *File \> Open File*

1.  **Launching from the Connection Dialog**

    Select the Occupancy Calculator activity from the connection dialog. You can optionally specify an occupancy calculator data file, which is used to initialize the calculator with the data from the saved file. Click the *Launch* button to open the Occupancy Calculator.

    

2.  **Launching from the Profiler Report**

    The Occupancy Calculator can be opened from the *Profiler Report* using the calculator button located in the report header or in the header of the *Occupancy* section on the *Detail Page*.

    
    Details page header

    
    Occupancy section header

The user interface consists of an input section as well as tables and graphs that display information about GPU occupancy. To use the calculator, change the input values in the input section, click the *Apply* button and examine the tables and graphs.

### 3.12.1. Tables

The tables show the occupancy, as well as the number of active threads, warps, and thread blocks per multiprocessor, and the maximum number of active blocks on the GPU.



Tables

### 3.12.2. Utilization

These stacked bar graphs show the portion of resources that is allocated to the blocks, the unallocated portion due to some other resource being a limiter, and lastly the unused portion that has the size less than the minimum amount required by a block.



Utilization

### 3.12.3. Graphs

The graphs show the occupancy for your chosen block size as a blue circle, and for all other possible block sizes as a line graph. The “Show As” option allows you to switch between the occupancy in percent and absolute number of warps on Y-axis. When you hover over the graphs, the nearest data point will be highlighted and its X and Y-axis values will appear in a tooltip.



Graphs

### 3.12.4. GPU Data

The *GPU Data* shows the properties of all supported devices.



GPU Data

## 3.13. Green Contexts support

NVIDIA Nsight Compute provides a number of features to ease the analysis of applications using [CUDA Green Contexts](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS), see the [Profiling Guide](../ProfilingGuide/index.html#special-configurations-green-contexts) for an overview about profiling such applications.

To identify Green Context results quickly, the icon of the *Attributes* button in the [Profiler Report Header](index.html#profiler-report-header) turns into a *leaf* for Green Context results. Clicking on this button navigates to the *Launch Statistics* section of the result (if available), which contains [additional metrics](../ProfilingGuide/index.html#id28) related to the Green Context (e.g., its ID and the number of SMs used). Some of this information is also available from the tooltip when hovering over the *Attributes* button.

To distinguish between Green-Contexts *attributable* and *non-attributable* metrics when navigating the [Details page](index.html#details-page), *Green Context markers* may be used. These can be toggled via the *View* menu of the [Report Header](index.html#profiler-report-header) and adds a *leaf* icon behind *attributable* metrics in all section headers. When hovering over attributable metrics, the tooltip also shows a hint about the scaling of such metrics. Additionally, the [Metrics Details](index.html#metric-details) tool window shows an *Attribution* row for metrics of Green Context results.



*Details* page of a Green Context result showing *Attributes* tooltip, *Metrics Details* tool window and Green Context markers.

If a report contains Green Context results, the [Session page](index.html#session-page) includes an additional *Green Context Resources* section. It contains a table with information about the resources used by each Green Context, such as the TPCs and workqueue resources assigned to it. The *Result IDs* column allows you to navigate to each profile result that uses the corresponding Green Context.



When profiling Green Context applications interactively, the [Resources](index.html#resources) tool window also allows to track the state of all Green Contexts, their resources, as well as their associated streams.



## 3.14. Acceleration Structure Viewer

The *Acceleration Structure Viewer* allows inspection of acceleration structures built using the OptiX API. In modern ray tracing APIs like OptiX, *acceleration structures* are data structures describing the rendered scene’s geometries that will be intersected when performing ray tracing operations. More information concerning acceleration structures can be found in the [OptiX programming guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html#acceleration_structures#acceleration-structures).

It is the responsibility of the user to set these up and pass them to the OptiX API which translates them to internal data structures that perform well on modern GPUs. The description created by the user can be very error-prone and it is sometimes hard to understand why the rendered result is not as expected. The *Acceleration Structure Viewer* is a component allowing OptiX users to inspect the acceleration structures they build before launching a ray tracing pipeline.

The *Acceleration Structure Viewer* is opened through a button in the [Resources](index.html#tool-window-resources) window. The button will only be available when the currently viewed resource is *OptiX: TraversableHandles*. It opens the currently selected handle.



The viewer is multi-paned: it shows a hierarchical view of the acceleration structure on the left, a graphical view of the acceleration structure in the middle, and controls and options on the right. In the hierarchical tree view on the left of the viewer the *instance acceleration structures (IAS)*, *geometry acceleration structures (GAS)*, child instances and child geometries are shown. In addition to this, some general properties for each of them is shown such as their primitive count, surface area and size on the device.



In the hierarchical view on the left of the *Acceleration Structure Viewer*, the following information is displayed where applicable.

| Column | Description |
|----|----|
| Name | An identifier for each row in the hierarchy. Click on the check box next to the name to show or hide the selected geometry or hierarchy. Double-click on this entry to jump to the item in the rendering view. |
| \# Prims | The number of primitives that make up this acceleration structure. |
| Surface Area | A calculation of the total surface area for the AABB that bounds the particular entry. |
| Size | The size of the output buffer on the device holding this *acceleration structure*. |

Acceleration Structure Hierarchical Columns {#id26 .table-no-stripes .docutils .align-default}

Performance analysis tools are accessible in the bottom left corner on the main view. These tools help identify potential performance problems that are outlined in the [RTX Ray Tracing Best Practices Guide](https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing). These analysis tools aim to give a broad picture of acceleration structures that may exhibit sub-optimal performance. To find the most optimal solution, profiling and experimentation is recommended but these tools may paint a better picture as to why one structure performs poorly compared to another.

| Action | Description |
|----|----|
| Instance Overlaps | Identifies instance AABBs that overlap with other instances in 3D. Consider merging GASes when instance world-space AABBs overlap significantly to potentially increase performance. |
| Instance Heatmap | This allows you to set the threshold used by the AABB heatmap rendered in the visualizer. |

Acceleration Structure Analysis Tools {#id27 .table-no-stripes .docutils .align-default}

### 3.14.1. Navigation

The *Acceleration Structure Viewer* supports multiple navigation modes. The navigation mode can be changed using the combo box in the camera controls pane, to the right of the rendering pane. The keyboard and mouse bindings for each mode are as follows:

| Binding | Fly Camera | Dolly Camera | Orbit Camera |
|----|----|----|----|
| **WASD/Arrow Keys** | Move forward, backward, left, right | Move forward, backward, left, right | Track (Move up, down, left, right) |
| **E/Q** | Move up/down | Move up/down | n/a |
| **Z/C** | Increase/decrease field of view | Increase/decrease field of view | Increase/decrease field of view |
| **Shift/Ctrl** | Move faster/slower | Move faster/slower | Move faster/slower |
| **Mousewheel** | Zoom in/out | Zoom in/out | Zoom in/out |
| **LMB + Drag** | Rotate in place | Rotate left/right, move forward/backward | Rotate around the geometry |
| **RMB + Drag** | Zoom in/out | Rotate in place | Zoom in/out |
| **MMB + Drag** | Track (Move up, down, left, right) | Track (Move up, down, left, right) | Track (Move up, down, left, right) |
| **Alt** | Temporarily switch to Orbit Camera | Temporarily switch to Orbit Camera | n/a |
| **F/Double Click** | Focus on the selected geometry | Focus on the selected geometry | Focus on the selected geometry |

Acceleration Structure Key Bindings {#id28 .table-no-stripes .docutils .align-default}

Based on the coordinate system of the input geometry, you may need to change the **Up Direction** setting to Z-Axis or the **Coordinates** setting to RHS. To reset the camera to its original location, click **Reset Camera**.

There are also a selection of Camera Controls for fast and precise navigation. To save a position, use the bookmarks controls. Each node within the acceleration structure hierarchy can also be double-clicked to quickly navigate to that location.



### 3.14.2. Filtering and Highlighting

The acceleration structure view supports acceleration structure filtering as well as highlighting of data matching particular characteristics. The checkboxes next to each geometry allow users to toggle the rendering of each traversable.

Geometry instances can also be selected by clicking on them in the main graphical view. Additionally, right clicking in the main graphical view gives options to hide or show all geometry, hide the selected geometry, or hide all but the selected geometry.



Beyond filtering, the view also supports highlight-based identification of geometry specified with particular flags. Checking each highlight option will identify those resources matching that flag, colorizing for easy identification. Clicking an entry in this section will dim all geometry that does **not** meet the filter criteria allowing items that match the filter to standout. Selecting multiple filters requires the passing geometry to meet all selected filters (e.g., AND logic). Additionally, the heading text will be updated to reflect the number of items that meet this filter criteria.



### 3.14.3. Rendering Options

Under the highlight controls, additional rendering options are available. These include methods to control the geometry colors and the ability to toggle the drawing of wireframes for meshes and AABBs.



### 3.14.4. Exporting

The data displayed in the acceleration structure viewer document can be saved to file. Exporting an *Acceleration Structure Viewer* document allows for persisting the data you have collected beyond the immediate analysis session. This capability is particularly valuable for comparing different revisions of your geometry or sharing with others. Bookmarks are persisted as well.

## 3.15. Options

NVIDIA Nsight Compute options can be accessed via the main menu under *Tools* \> *Options*. All options are persisted on disk and available the next time NVIDIA Nsight Compute is launched. When an option is changed from its default setting, its label will become bold. You can use the *Restore Defaults* button to restore all options to their default values.



Profile options

### 3.15.1. Profile

#### Report Sections

Configure the directories and import settings for section files and rules.

| Option Name | Description | Values |
|----|----|----|
| Sections Directory | Directory from which to import section files and rules. Relative paths are with respect to the NVIDIA Nsight Compute installation directory. |  |
| Include Sub- Directories | Recursively include section files and rules from sub-directories. | Yes (Default)/No |

NVIDIA Nsight Compute Report Section Options {#id29 .table-no-stripes .docutils .align-default}

#### Report Rules

Configure how rules are applied and reloaded during profiling sessions.

| Option Name | Description | Values |
|----|----|----|
| Apply Applicable Rules Automatically | Automatically apply active and applicable rules. | Yes (Default)/No |
| Reload Rules Before Applying | Force a rule reload before applying the rule to ensure changes in the rule script are recognized. | Yes/No (Default) |

NVIDIA Nsight Compute Report Rules Options {#id30 .table-no-stripes .docutils .align-default}

#### Report UI

Customize the user interface behavior for report viewing and navigation.

[TABLE]

NVIDIA Nsight Compute Report UI Options {#id31 .table-no-stripes .docutils .align-default}

#### Report Baselines

Configure baseline display settings for result comparison.

| Option Name | Description | Values |
|----|----|----|
| Maximum Baseline Name Length | The maximum length of baseline names. | 1..N (Default: 40) |
| Number of Full Baselines to Display | Number of baselines to display in the report header with all details in addition to the current result or the baseline added for the current result. | 0..N (Default: 2) |

NVIDIA Nsight Compute Report Baselines Options {#id32 .table-no-stripes .docutils .align-default}

#### Report Metrics

Configure how metrics are displayed and processed in reports.

| Option Name | Description | Values |
|----|----|----|
| Auto-Convert Metric Units | Auto-adjust displayed metric units and values (e.g. Bytes to KBytes). | Yes (Default)/No |
| Show Instanced Metric Values | Show the individual values of instanced metrics in tables. | Yes/No (Default) |
| Show Metrics As Floating Point | Show all numeric metrics as floating-point numbers. | Yes/No (Default) |
| Show Knowledge Base Information | Show information from the knowledge base in (metric) tooltips to explain terminology. Note: Nsight Compute needs to be restarted for this option to take effect. | Yes (Default)/No |

NVIDIA Nsight Compute Report Metrics Options {#id33 .table-no-stripes .docutils .align-default}

#### Report Summary Page

Configure settings for the Summary page in reports.

| Option Name | Description | Values |
|----|----|----|
| Metrics/ Properties | List of metrics and properties to show on the summary page. Comma-separated list of metric entries. Each entry has the format {Label:MetricName}. |  |

NVIDIA Nsight Compute Report Summary Page Options {#id34 .table-no-stripes .docutils .align-default}

#### Report Details Page

Configure settings for the Details page in reports.

| Option Name | Description | Values |
|----|----|----|
| Default Section Body Visibility | Controls the default visibility of section bodies when opening a report. | Default (Default)/All |

NVIDIA Nsight Compute Report Summary Page Options {#id35 .table-no-stripes .docutils .align-default}

#### Report Source Page

Configure settings for the Source page view.

| Option Name | Description | Values |
|----|----|----|
| Delay Load ‘Source’ Page | Delays loading the content of the report page until the page becomes visible. Avoids processing costs and memory overhead until the report page is opened. | Yes/No (Default) |
| Show Single File For Multi-File Sources | Shows a single file in each Source page view, even for multi-file sources. | Yes/No (Default) |
| Show Only Executed Functions | Shows only executed functions in the source page views. Disabling this can impact performance. | Yes (Default)/No |
| Default Metric Value Mode | Default setting for the mode in which the metric values are displayed. | Relative (Default)/Absolute |
| Default Metric Precision Mode | Default setting for the precision in which absolute metric values are displayed. | Abbreviated (Default)/Full |
| Auto-Resolve Remote Source Files | Automatically try to resolve remote source files on the source page (e.g. via SSH) if the connection is still registered. | Yes/No (Default) |
| Enable Register Dependencies | Track dependencies between SASS registers/predicates and display them in the SASS view. | Yes (Default)/No |
| SASS Analysis Size Threshold (KB) | Enable SASS flow graph analysis for functions below this threshold. SASS analysis is required for Live Register and Register Dependency information. Set to -1 to enable analysis for all functions. | -1..N (Default: 1024) |
| Enable ELF Verification | Enable ELF (cubin) verification to run every time before SASS analysis. This should only be enabled when working with applications compiled before CUDA 11.0 or when encountering source page issues. | Yes/No (Default) |

NVIDIA Nsight Compute Source Page Options {#id36 .table-no-stripes .docutils .align-default}

#### API Stream View

Configure settings for the API Stream View.

| Option Name | Description | Values |
|----|----|----|
| API Call History | Number of recent API calls shown in API Stream View. | 1..N (Default: 100) |

NVIDIA Nsight Compute API Stream View Options {#id37 .table-no-stripes .docutils .align-default}

#### CUDA Graph Viewer

Configure settings for the CUDA Graph Viewer.

| Option Name | Description | Values |
|----|----|----|
| Auto-Open Graph Viewer | Automatically open the CUDA Graph Viewer when CUDA graphs are detected during profiling or when opening reports containing CUDA graph results. | Yes (Default)/No |

NVIDIA Nsight Compute CUDA Graph Viewer Options {#id38 .table-no-stripes .docutils .align-default}

### 3.15.2. Fonts and Colors

#### Fonts

Configure the fonts used throughout the application interface. General fonts affect all UI elements except code, while code fonts are specifically used for source code display and similar content.

| Option Name | Description | Values |
|----|----|----|
| General Font | General font used for all non-code UI elements. | Select an installed font via “Change…” button. (Default: Roboto@9pt) |
| Code Font | Font used for source code and code-like UI elements. | Select an installed font via “Change…” button. (Default: Cascadia Mono@9pt) |

NVIDIA Nsight Compute Font Options {#id39 .table-no-stripes .docutils .align-default}

#### Colors

Customize the visual appearance of the application through theme selection and color scale preferences. These settings affect the overall look and feel of the interface and how data is represented visually.

[TABLE]

NVIDIA Nsight Compute Color Options {#id40 .table-no-stripes .docutils .align-default}

### 3.15.3. Environment

#### Visual Experience

Configure how NVIDIA Nsight Compute handles display and windowing behaviors, particularly for multi-monitor setups.

| Option Name | Description | Values |
|----|----|----|
| Use Enhanced Windowing Experience | Disable Mixed DPI Scaling if unwanted artifacts are detected when using monitors with different DPIs. Requires app restart. | Yes (Default)/No |

NVIDIA Nsight Compute Visual Experience Options {#id41 .table-no-stripes .docutils .align-default}

#### Windowing

Control window behavior and management settings.

| Option Name | Description | Values |
|----|----|----|
| Floating Windows Always on Top | Configure floating tool windows to always stay on top of the main window. Requires app restart. | Yes/No (Default) |

NVIDIA Nsight Compute Windowing Options {#id42 .table-no-stripes .docutils .align-default}

#### Documents Folder

Configure where NVIDIA Nsight Compute stores project files and documentation.

| Option Name | Description | Values |
|----|----|----|
| Documents Folder | The folder where projects and documents will be saved. |  |

NVIDIA Nsight Compute Documents Folder Options {#id43 .table-no-stripes .docutils .align-default}

#### Startup Behavior

Define how NVIDIA Nsight Compute behaves when launched.

[TABLE]

NVIDIA Nsight Compute Startup Behavior Options {#id44 .table-no-stripes .docutils .align-default}

#### Updates

Configure version update notification settings.

| Option Name | Description | Values |
|----|----|----|
| Show version update notifications | Show notifications when a new version of this product is available. | Yes (Default)/No |

NVIDIA Nsight Compute Updates Options {#id45 .table-no-stripes .docutils .align-default}

### 3.15.4. Connection

Connection properties are grouped into *Target Connection Options* and *Host Connection Properties*.

#### Target Connection Properties

The *Target Connection Properties* determine how the host connects to the target application during an *Interactive Profile Activity*. This connection is used to transfer profile information to the host during the profile session.

| Option Name | Description | Values |
|----|----|----|
| Base Port | Base port used to establish a connection from the host to the target application during an *Interactive Profile* activity (both local and remote). | 1-65535 (Default: 49152) |
| Maximum Ports | Maximum number of ports to try (starting from *Base Port*) when attempting to connect to the target application. | 2-65534 (Default: 64) |

NVIDIA Nsight Compute Target Connection Properties {#id46 .table-no-stripes .docutils .align-default}

#### Host Connection Properties

The *Host Connection Properties* determine how the command line profiler will connect to the host application during a *Profile Activity*. This connection is used to transfer profile information to the host during the profile session.

| Option Name | Description | Values |
|----|----|----|
| Base Port | Base port used to establish a connection from the command line profiler to the host application during a *Profile* activity (both local and remote). | 1-65535 (Default: 50152) |
| Maximum Ports | Maximum number of ports to try (starting from *Base Port*) when attempting to connect to the host application. | 1-100 (Default: 10) |

NVIDIA Nsight Compute Host Connection Options {#id47 .table-no-stripes .docutils .align-default}

#### SSH ProxyJump Connection Properties

The *SSH ProxyJump Connection Properties* configure how NVIDIA Nsight Compute handles connections through SSH jump hosts. This is useful when profiling applications on target machines that are not directly accessible and require connecting through intermediate SSH servers. The ProxyJump feature allows for secure, multi-hop SSH connections while maintaining the profiling functionality.

| Option Name | Description | Values |
|----|----|----|
| Process Scan Timeout (seconds) | Maximum time to try searching for attachable process when using SSH ProxyJump connection. | 1-65535 (Default: 10) |

NVIDIA Nsight Compute SSH ProxyJump Connection Options {#id48 .table-no-stripes .docutils .align-default}

### 3.15.5. Timeline

#### Basic Settings

The *Timeline Basic Settings* control the visualization and behavior of the Timeline view in NVIDIA Nsight Compute.

| Option Name | Description | Values |
|----|----|----|
| Show correlation arrows | Show arrows on correlated items on the Timeline | Yes (Default)/No |

NVIDIA Nsight Compute Timeline Options {#id49 .table-no-stripes .docutils .align-default}

### 3.15.6. Source Lookup

The Source Lookup options control how NVIDIA Nsight Compute locates and validates CUDA source files when displaying source code in the Source page. These settings are particularly important for debugging and profiling sessions where source files may have moved from their original compilation locations or when working with different development environments. The options help ensure accurate source code resolution and provide flexibility in handling file location mismatches.

| Option Name | Description | Values |
|----|----|----|
| Program Source Locations | Set program source search paths. These paths are used to resolve CUDA-C source files on the Source page if the respective file cannot be found in its original location. Files which cannot be found are marked with a *File Not Found* error. See the *Ignore File Properties* option for files that are found but don’t match. |  |
| Ignore File Properties | Ignore file properties (e.g. timestamp, size) for source resolution. If this is disabled, all file properties like modification timestamp and file size are checked against the information stored by the compiler in the application during compilation. If a file with the same name exists on a source lookup path, but not all properties match, it won’t be used for resolution (and a *File Mismatch* error will be shown). | Yes/No (Default) |

NVIDIA Nsight Compute Source Lookup Options {#id50 .table-no-stripes .docutils .align-default}

### 3.15.7. Send Feedback…

| Option Name | Description | Values |
|----|----|----|
| Collect Usage and Platform Data | Choose whether or not you wish to allow NVIDIA Nsight Compute to collect usage and platform data. | Yes (Default)/No |

NVIDIA Nsight Compute Send Feedback Options {#id51 .table-no-stripes .docutils .align-default}

## 3.16. Projects

NVIDIA Nsight Compute uses *Project Files* to group and organize profiling reports. At any given time, only one project can be open in NVIDIA Nsight Compute. Collected reports are automatically assigned to the current project. Reports stored on disk can be assigned to a project at any time. In addition to profiling reports, related files such as notes or source code can be associated with the project for future reference.

Note that only references to reports or other files are saved in the project file. Those references can become invalid, for example when associated files are deleted, removed or not available on the current system, in case the project file was moved itself.

NVIDIA Nsight Compute uses the `ncu-proj` file extension for project files.

When no custom project is current, a *default project* is used to store e.g. the current [Start Activity Dialog](index.html#connection-dialog) entries. To remove all information from the default project, you must close NVIDIA Nsight Compute and then delete the file from disk.

- On Windows, the file is located at `<USER>\AppData\Local\NVIDIA`` ``Corporation\NVIDIA`` ``Nsight`` ``Compute\`

- On Linux, the file is located at `<USER>/.local/share/NVIDIA`` ``Corporation/NVIDIA`` ``Nsight`` ``Compute/`

- On macOS, the file is located at `<USER>/Library/Application`` ``Support/NVIDIA`` ``Corporation/NVIDIA`` ``Nsight`` ``Compute/`

### 3.16.1. Project Dialogs

**New Project**

Creates a new project. The project must be given a name, which will also be used for the project file. You can select the location where the project file should be saved on disk. Select whether a new directory with the project name should be created in that location.

### 3.16.2. Project Explorer

The *Project Explorer* window allows you to inspect and manage the current project. It shows the project name as well as all *Items* (profile reports and other files) associated with it. Right-click on any entry to see further actions, such as adding, removing or grouping items. Type in the *Search project* toolbar at the top to filter the currently shown entries.



Project Explorer

## 3.17. Visual Studio Integration Guide

This guide provides information on using NVIDIA Nsight Compute within Microsoft Visual Studio, using the [NVIDIA Nsight Integration](https://developer.nvidia.com/nsight-tools-visual-studio-integration) Visual Studio extension, allowing for a seamless development workflow.

### 3.17.1. Visual Studio Integration Overview

NVIDIA Nsight Integration is a Visual Studio extension that allows you to access the power of NVIDIA Nsight Compute from within Visual Studio.

When NVIDIA Nsight Compute is installed along with NVIDIA Nsight Integration, NVIDIA Nsight Compute activities will appear under the NVIDIA ‘Nsight’ menu in the Visual Studio menu bar. These activities launch NVIDIA Nsight Compute with the current project settings and executable.

For more information about using NVIDIA Nsight Compute from within Visual Studio, please visit

- [NVIDIA Nsight Integration Overview](https://developer.nvidia.com/nsight-tools-visual-studio-integration)

- [NVIDIA Nsight Integration User Guide](https://docs.nvidia.com/nsight-vs-integration/index.html)

Notices

Notices

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

Information furnished is believed to be accurate and reliable. However, NVIDIA Corporation assumes no responsibility for the consequences of use of such information or for any infringement of patents or other rights of third parties that may result from its use. No license is granted by implication of otherwise under any patent rights of NVIDIA Corporation. Specifications mentioned in this publication are subject to change without notice. This publication supersedes and replaces all other information previously supplied. NVIDIA Corporation products are not authorized as critical components in life support devices or systems without express written approval of NVIDIA Corporation.

Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.


---


<!-- source: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html -->
# 4. Nsight Compute CLI

The User Guide for Nsight Compute CLI.

## 4.1. Introduction

NVIDIA Nsight Compute CLI (ncu) provides a non-interactive way to profile applications from the command line. It can print the results directly on the command line or store them in a report file. It can also be used to simply launch the target application (see [General](index.html#command-line-options-general) for details) and later attach with NVIDIA Nsight Compute or another ncu instance.

## 4.2. Quickstart

1.  **Launch the target application with the command line profiler**

    The command line profiler launches the target application, instruments the target API, and collects profile results for the specified kernels. The CLI executable is called ncu. A shortcut with this name is located in the base directory of the NVIDIA Nsight Compute installation. The actual executable is located in the folder `target\windows-desktop-win7-x64` on Windows or `target/linux-desktop-glibc_2_11_3-x64` on Linux. By default, NVIDIA Nsight Compute is installed in `/usr/local/cuda-<cuda-version>/NsightCompute-<version>` on Linux and in `C:\Program`` ``Files\NVIDIA`` ``Corporation\Nsight`` ``Compute`` ``<version>` on Windows.

    To collect the `basic` set for all kernel launches in the target application, launch:

    ``` text
    $ ncu -o profile CuVectorAddMulti.exe
    ```

    The application runs in instrumented mode and for each kernel launch, a profile result is created. The results are written by default to profile.nsight-cuprof. Each output from the compute profiler starts with `==PROF==` The other lines are output from the application itself. For each profiled kernel, the name of the kernel function and the progress of data collection is shown. To collect all requested profile information, it may be required to replay the kernels multiple times. The total number of replay passes per kernel is shown after profiling has completed.

    ``` text
    [Vector addition of 1144477 elements]
    ==PROF== Connected to process 5268
    Copy input data from the host memory to the CUDA device
    CUDA kernel launch A with 4471 blocks of 256 threads
    ==PROF== Profiling "vectorAdd_A" - 0: 0%....50%....100% - 46 passes
    CUDA kernel launch B with 4471 blocks of 256 threads
    ==PROF== Profiling "vectorAdd_B" - 1: 0%....50%....100% - 46 passes
    Copy output data from the CUDA device to the host memory
    Done
    ==PROF== Disconnected from process 5268
    ==PROF== Report: profile.ncu-rep
    ```

2.  **Customizing data collection**

    Options are available to specify for which kernels data should be collected. `-c` limits the number of kernel launches collected. `-s` skips the given number of kernels before data collection starts. `-k` allows you to filter the kernels by a regex match of their names. `--kernel-id` allows you to filter kernels by context, stream, name and invocation.

    To limit what should be collected for each kernel launch, specify the exact \*.section (files) by their identifier using `--section`. Each section file defines a set of metrics to be collected, grouped logically to solve a specific performance question. By default, the sections associated with the `basic` set are collected. Use `--list-sets` to see the list of currently available sets. Use `--list-sections` to see the list of currently available sections. The default search directory and location of pre-defined section files is also called `sections/`. See the [Profiling Guide](../ProfilingGuide/index.html#sets-and-sections) for more details.

    Alternatively, you can collect a set of individual metrics using `--metrics`. The available metrics can be queried using `--query-metrics`. For an explanation of the naming conventions and structuring of metrics, see [Metrics Structure](../ProfilingGuide/index.html#metrics-structure).

    Most metrics in NVIDIA Nsight Compute are named using a base name and various suffixes, e.g. *sm\_\_throughput.avg.pct_of_peak_sustained_elapsed*. The base name is *sm\_\_throughput and the suffix is avg.pct_of_peak_sustained_elapsed*. This is because most metrics follow the same structure and have the same set of suffixes. You need to pass the base or full name to NVIDIA Nsight Compute when selecting a metric for profiling. Use `--query-metrics-mode`` ``suffix`` ``--metrics`` ``<metrics`` ``list>` to see the full names for the chosen metrics.

    Some additional metrics do not follow this structured naming. They are documented in the [Metrics Reference](../ProfilingGuide/index.html#metrics-reference).

3.  **Changing command line output**

    By default, a temporary file is used to store profiling results, and data is printed to the command line. To permanently store the profiler report, use `-o` to specify the output filename.

    Besides storing results in a report file, the command line profiler can print results using different pages. Those pages correspond to the respective pages in the UI’s report. By default, the [Details page](index.html#output-pages) is printed, if no explicit output file is specified. To select a different page or print in addition to storing in an explicit file, use the `--page=<Page>` command. Currently, the following pages are supported: `details,`` ``raw,`` ``source`.

    Use `--csv` to make any output comma separated and easier to process further. See [Console Output](index.html#command-line-options-console-output) for further options, e.g. summary views.

4.  **Open the report in the UI**

    The UI executable is called ncu-ui. A shortcut with this name is located in the base directory of the NVIDIA Nsight Compute installation. The actual executable is located in the folder `host\windows-desktop-win7-x64` on Windows or `host/linux-desktop-glibc_2_11_3-x64` on Linux. In the UI window, close the *Start Activity* dialog and open the report file through *File* \> *Open*, by dragging the report file into NVIDIA Nsight Compute.

    You can also specify the report file as a command line parameter to the executable, i.e. as `ncu-ui`` ``<MyReport.ncu-rep>`. Alternatively, when using NVIDIA Nsight Compute CLI on a platform with host support, `--open-in-ui` can be used directly with ncu to open a collected report in the user interface.

    The report opens in a new document window. For more information about the report, see the [Profiler Report](../NsightCompute/index.html#profiler-report) for collecting profile information through NVIDIA Nsight Compute.

## 4.3. Usage

### 4.3.1. Modes

Modes change the fundamental behavior of the command line profiler. Depending on which mode is chosen, different [Command Line Options](index.html#command-line-options) become available. For example, [Launch](index.html#command-line-options-launch) is invalid if the *Attach* mode is selected.

- **Launch-and-attach:** The target application is launched on the local system with the tool’s injection libraries. Depending on which profiling options are chosen, selected kernels in the application are profiled and the results printed to the console or stored in a report file. The tool exits once the target application finishes or crashes, and once all results are processed.

  This is the default, and the only mode that supports profiling of child processes on selected platforms.

- **Launch:** The target application is launched on the local system with the tool’s injection libraries. As soon as the first intercepted API call is reached (commonly `cuInit()`), all application threads are suspended. The application now expects a tool to attach for profiling. You can attach using NVIDIA Nsight Compute or using the command line profiler’s *Attach* mode.

- **Attach:** The tool tries to connect to a target application previously launched using NVIDIA Nsight Compute or using the command line profiler’s *Launch* mode. The tool can attach to a target on the local system or using a remote connection.

### 4.3.2. Multi-Process Support

The NVIDIA Nsight Compute CLI supports profiling multi-process applications. See [Launch](index.html#command-line-options-launch) for which command line options are available.

#### Platform Support

On x86_64 Windows, NVIDIA Nsight Compute CLI supports profiling 64-bit processes launched from 32-bit applications by default. On x86_64 Linux, launching from 32-bit applications requires you to enable the `support-32bit` option, and the required 32-bit libraries must be installed on your system. On DRIVE OS Linux and DRIVE OS QNX, tracking of 32-bit applications is not supported. Profiling of 32-bit processes is not supported on any platform.

#### MPI Support

NVIDIA Nsight Compute CLI can be used to profile applications launched with the `mpirun` command.

- To profile all ranks on a node and store all the profiling data in a single report file:

  ``` cpp
  ncu --target-processes all -o <report-name> mpirun [mpi arguments] <app> [app arguments]
  ```

- To profile multi-node submissions, ensure that you specify unique report files per rank.

  ``` cpp
  mpirun [mpi arguments] ncu -o report_%q{OMPI_COMM_WORLD_RANK} <app> [app arguments]
  ```

  Alternatively, enable the multiprocess communicator:

  ``` cpp
  mpirun [mpi arguments] ncu --communicator=tcp --communicator-num-peers=<number of MPI ranks> -o report <app> [app arguments]
  ```

- To profile a single rank one can use a wrapper script. The following script (called “wrap.sh”) profiles rank 0 only:

  ``` cpp
  #!/bin/bash
  if [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
     ncu -o report_${OMPI_COMM_WORLD_RANK}  --target-processes all "$@"
  else
     "$@"
  fi
  ```

  and then execute:

  ``` cpp
  mpirun [mpi arguments] ./wrap.sh <app> [app arguments]
  ```

See below for profiling mandatory concurrent kernels across MPI processes.

#### Mandatory Concurrent Kernels

Multi-process/multi-GPU applications that contain mandatory concurrent kernels (e.g., NCCL and NVSHMEM communication kernels) may be unable to make forward progress when profiled with default settings due to multi-pass data collection and serialization across processes within the same process tree.

**For communication kernels launched from different process trees, use --communicator tcp**

To make the tool synchronize launchs and replays across processes, use `--lockstep-kernel-launch`.

> ``` cpp
> mpirun -np 4 ncu --communicator tcp --communicator-num-peers 4 --lockstep-kernel-launch -o report <app>
> ```

Additionally, `--lockstep-nvtx-include` and `--lockstep-nvtx-exclude` options can be set to limit synchronization to specific kernel ranges. See [NVTX Filtering](index.html#nvtx-filtering) for additional information.

> ``` cpp
> mpirun <args> ncu <args> --lockstep-kernel-launch --lockstep-nvtx-include nccl/ -o report <app>
> ```

[--communicator tcp](index.html#launch) internally uses TCP sockets in a client-server architecture to send and receive message. One process at random is selected as the server process to manage sockets. In a multi-node system, the option `--communicator-tcp-hostname` must be set to manually select one host as the server. Below is an example slurm script that can be used to automate the process

> ``` cpp
> #!/bin/bash
> nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
> nodes_array=($nodes)
> head_node=${nodes_array[0]}
> head_node_addr=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname)
>
> srun ncu \
>     --communicator=tcp                           \
>     --communicator-tcp-hostname="$head_node_addr"  \
>     --communicator-tcp-num-peers="$SLURM_NTASKS" \
>     --lockstep-kernel-launch                     \
>     ./application
> ```

You can use the [Report Merge Tool](../NsightCompute/index.html#report-merge-tool) and [Clustering Window](../NsightCompute/index.html#clustering-window) to aggregate and/or compare these results.

**For communication kernels launched from the same process tree, use --communicator shmem**

In single-node scenarios where the entire process tree is profiled by the same instance of `ncu`, the [--communicator shmem](index.html#launch) option can be used. The processes must be able to communicate with each other using shared memory on the target system.

> ``` cpp
> ncu --communicator shmem --communicator-shmem-num-peers 2 -f -o report -k regex:nccl.* torchrun --nnodes=1 --nproc_per_node=2 pytorch_app.py
> ```

This mode only supports *kernel* and *range* replay modes and has limited filter options. If your app launches different types of kernels, select the concurrent ones using a name filter like `-k`` ``regex:nccl.*`. A single report is generated for all profile results. By default, the first CUDA context of each group is profiled. The `--devices` option can be used to select the primary GPU device(s) to profile.

Setting `--communicator-shmem-num-peers` to the number of processes participating in the concurrent workload `-k`` ``...` is not required but recommended. If the number of processes is not specified, the communicator will use all available processes per workload. If processes don’t reach the same workload at the same time, and since the number of processes per workload is unknown to the tool, Nsight Compute may try to profile a workload with insufficient participants.

Launcher processes like `mpirun` or `torchrun` are auto-excluded from profiling. You can manually exclude others using the [–target-processes-filter](index.html#launch) option.

### 4.3.3. Output Pages

The command line profiler supports printing results to the console using various pages. Each page has an equivalent in NVIDIA Nsight Compute’s *Profiler Report*. In the command line profiler, they are slightly adapted to fit console output. To select a page, use the `--page` option. By default, the details page is used. Note that if `--page` is not used but `--export` is, no results will be printed to the console.

- **Details:** This page represents NVIDIA Nsight Compute’s *Details* page. For every profiled kernel launch, each collected is printed as section as a three-column table, followed by any rule results applied to this section. Rule results not associated with any section are printed after the kernel’s sections.

  The first section table column shows the metric name. If the metric was given a label in the section, it is used instead. The second column shows the metric unit, if available. The third column shows the unit value. Both metric unit and value are automatically adjusted to the most fitting order of magnitude. By default, only metrics defined in section headers are shown. This can be changed by passing the `--details-all` option on the command line.

  Some metrics will show multiple values, separated by “;”, e.g. memory_l2_transactions_global Kbytes 240; 240; 240; 240; 240. Those are instanced metrics, which have one value per represented instance. An instance can be a streaming multiprocessor, an assembly source line, etc.

- **Raw:** This page represents NVIDIA Nsight Compute’s *Raw* page. For every profiled kernel launch, each collected metric is printed as a three-column table. Besides metrics from sections, this includes automatically collected metrics such as device attributes and kernel launch information.

  The first column shows the metric name. The second and third columns show the metric unit and value, respectively. Both metric unit and value are automatically adjusted to the most fitting order of magnitude. No unresolved regex:, group:, or breakdown: metrics are included.

### 4.3.4. Profile Import

Using the `--import` option, saved reports can be imported into the command line profiler. When using this flag, most other options are not available, except for certain result filterting options. They are marked as such in the [Profile options](index.html#command-line-options-profile) table.

### 4.3.5. Filtered Profile Export

Using the `--import` and `--export` options together, along with supported filtering options, you can export desired results from one report to another. Most of the filtering [Profile options](index.html#command-line-options-profile) that can be used with `--import` alone are also supported here, except for `--metrics` and `--section`.

### 4.3.6. Metrics and Units

When available and applicable, metrics are shown along with their unit. This is to make it apparent if a metric represents cycles, threads, bytes/s, and so on.

By default, units are scaled automatically so that metric values are shown with a reasonable order of magnitude. Units are scaled using their SI-factors, i.e. byte-based units are scaled using a factor of 1000 and the prefixes K, M, G, etc. Time-based units are also scaled using a factor of 1000, with the prefixes n, u and m. This scaling can be changed using a command line option, see [Console Output](index.html#command-line-options-console-output) options for details.

### 4.3.7. NVTX Filtering

`--nvtx-include`` ``<configuration>`` ``--nvtx-exclude`` ``<configuration>`

These options are used to profile only those kernels which satisfy the conditions mentioned in the configuration. Through these options, you can choose which kernel falls into a specific range or collection of ranges.

You can use both options multiple times, mentioning all the `--nvtx-include` configurations followed by all `--nvtx-exclude` configurations. NVTX filtering requires `--nvtx` option.

NVTX ranges are of two types: NvtxRangeStart/End and NvtxRangePush/Pop. The configuration syntax for both the types are briefly described below. Both range and domain names can contain whitespace. Note that “Domain” and “range” in below example are for illustration purposes only and are not required to mark domain or range names.

- **Push-Pop Ranges**

  [TABLE]

  Include kernels wrapped inside push/pop range ‘A_range’ of ‘\<default-domain\>’:

  ``` text
  ncu --nvtx --nvtx-include "A_range/" CuNvtx.exe
  ```

  Include kernels wrapped inside push/pop range ‘A_range’ of ‘Domain-A’:

  ``` text
  ncu --nvtx --nvtx-include "Domain-A@A_range/" CuNvtx.exe
  ```

  Include kernels wrapped inside push/pop range ‘A_range’ of ‘\<default domain\>’, where ‘A_range’ is at the bottom of the stack:

  ``` text
  ncu --nvtx --nvtx-include "[A_range" CuNvtx.exe
  ```

  Include kernels wrapped inside push/pop ranges ‘A_range’ and ‘B range’ of ‘\<default domain\>’, with zero or many ranges between them:

  ``` text
  ncu --nvtx --nvtx-include "A_range/*/B range" CuNvtx.exe
  ```

  Exclude kernels wrapped inside push/pop ranges ‘A_range’ and ‘B range’ of ‘\<default domain\>’, with zero or many ranges between them:

  ``` text
  ncu --nvtx --nvtx-exclude "A_range/*/B range" CuNvtx.exe
  ```

  Include kernels wrapped inside only push/pop range ‘A_range’ of ‘\<default domain\>’ but not inside ‘B range’ at the top of the stack:

  ``` text
  ncu --nvtx --nvtx-include "A_range/" --nvtx-exclude "B range]" CuNvtx.exe
  ```

- **Start-End Ranges**

  [TABLE]

  Include kernels wrapped inside start/end range ‘A_range’ of ‘Domain-A’:

  ``` text
  ncu --nvtx --nvtx-include "Domain-A@A_range" CuNvtx.exe
  ```

  Include kernels wrapped inside both start/end ranges, ‘A_range’ and ‘B range’ of ‘\<default domain\>’:

  ``` text
  ncu --nvtx --nvtx-include "A_range,B range" CuNvtx.exe
  ```

  Include kernels wrapped inside start/end ranges, ‘A_range’ or ‘B range’ of ‘\<default domain\>’:

  ``` text
  ncu --nvtx --nvtx-include "A_range" --nvtx-include "B range" CuNvtx.exe
  ```

  Include all kernels, except those which are wrapped inside start/end range ‘A_range’ of ‘\<default domain\>’:

  ``` text
  ncu --nvtx --nvtx-exclude "A_range" CuNvtx.exe
  ```

  Include kernels wrapped inside only start/end ‘B range’ and not ‘A_range’ of ‘\<default domain\>’:

  ``` text
  ncu --nvtx --nvtx-include "B range"--nvtx-exclude "A_range" CuNvtx.exe
  ```

- **Regular Expression Support**

  The configuration syntax for both the types NvtxRangeStart/End and NvtxRangePush/Pop is the same. Additionally, to use regular expressions, follow the following syntax.

  - Provide prefix ‘regex:’ to treat nvtx config as regular expression.

    ``` text
    ncu --nvtx --nvtx-include "regex:Domain[A-Z]@Range[0-9]/" CuNvtx.exe
    ```

    The kernels wrapped inside push/pop range with matching regex ‘Range\[0-9\]’ of domain with matching regex ‘Domain\[A-Z\]’ are profiled.

  - Provide ‘/’ as a prefix to “\[” or “\]” only for the range part of the config if “\[” or “\]” is at the start or at the end of the range part, respectively. This is needed so that NCU can distinguish if “\[” or “\]” is part of the regex or represents the top/bottom of the stack.

    ``` text
    ncu --nvtx --nvtx-include "regex:[0-9]domainA@/[0-9]rangeA,RangeC[0-9/]" CuNvtx.exe
    ```

    The kernels wrapped inside start/end ranges with matching regex ‘\[0-9\]rangeA’ and ‘RangeC\[0-9\]’ of domain with matching regex ‘\[0-9\]domainA’ are profiled.

  - If any quantifier is part of the domain/range name, you need to use ‘\\’ or ‘\\ as a prefix. For the “\$” quantifier, only the ‘\\’ prefix is valid.

- **Additional Information**

  ``` text
  --nvtx-include "DomainA@RangeA,DomainB@RangeB" //Not a valid config
  ```

  In a single NVTX configuration, multiple ranges with regard to a single domain can be specified. Mentioning ranges from different domains inside a single NVTX config is not supported.

  ``` text
  --nvtx-include "A_range\[i\]"
  ```

  Quantifiers `@`` ``,`` ``[`` ``]`` ``/`` ``*`` ``+`, including regular expression quantifiers, can be used in domain/range names using prefix ‘\\. The kernels wrapped inside ‘A_range\[i\]’ of ‘\<default domain\>’ in the application are profiled.

  ``` text
  --nvtx-include "A_range"  //Start/End configuration
  --nvtx-inlcude "A_range/" //Push/Pop configuration
  --nvtx-inlcude "A_range]" //Push/Pop configuration
  ```

  If the domain/range name contains ‘\\, you need to provide ‘\\\\’ in the config.

  Do not use ‘\\\\’ before any quantifier.

  Including/Excluding only single range for Push/Pop configuration without specifying stack frame position ‘\[’ or ‘\]’, use ‘/’ quantifier at the end.

  ``` text
  --nvtx-include "A_range/*/B range"
  ```

  The order in which you mention Push/Pop configurations is important. In the above example, ‘A_range’ should be below ‘B range’ in the stack of ranges so that the kernel is profiled.

  NVTX filtering honors cudaProfilerStart() and cudaProfilerStop(). There is no support for ranges with no name.

### 4.3.8. Config File

Using the `--config-file`` ``on/off` option, parsing parameters from config file can be enabled or disabled.

Using the `--config-file-path`` ``<path>` option, default path and name of config file can be overwritten.

By default, config-file with name `config.ncu-cfg` is searched in the current working directory, `$HOME/.config/NVIDIA`` ``Corporation` on Linux and `%APPDATA%\NVIDIA`` ``Corporation\` on Windows. If a valid config file is found, ncu parses the file and initializes any command line parameters to the values set in the file. If the same command line parameter is also set explicitly during the current invocation, the latter takes precedence.

Parameters can be set under various general modes and ncu command line parameters are used to determine which general-mode needs to be parsed from the config file. See the table below for more details.

| Command line parameters | General Mode |
|----|----|
| ncu –mode launch-and-attach CuVectorAddMulti.exe | Launch-and-attach |
| ncu –mode launch CuVectorAddMulti.exe | Launch |
| ncu –mode attach | Attach |
| ncu –list-sets, ncu –list-sections, ncu –list-rules and ncu –list-metrics | List |
| ncu –query-metrics | Query |
| ncu -i \<MyReport.ncu-rep\> | Import |

These general modes should be defined in the config file using INI-like syntax as:

``` text
[<general-mode>]
<parameter>=<value>
;<comments>
```

**Sample usage**

``` text
[Launch-and-attach]
-c = 1
--section = LaunchStats, Occupancy
[Import]
--open-in-ui
-c = 1
--section = LaunchStats, Occupancy
```

From this configuration, ncu will parse parameters set under `[Launch-and-attach]` block whenever an application is profiled in `launch-and-attach` mode. In the same manner, parameters set under `[Import]` block will be parsed whenever a report is imported. Different modes can be clubbed together if there exists a set of parameters which is common to each mode. Sample shown above can be rewritten after clubbing both modes as:

``` text
[Launch-and-attach, import]
-c = 1
--section = LaunchStats, Occupancy
[Import]
--open-in-ui
```

**Additional points**

- Options like `--open-in-ui` do not expect any value to be set. These options should not be passed any value.

- Options like `--section` can be passed multiple times in the command line. These options should be written only once under a general-mode with all required values seperated by comma as shown below. Explicitly setting values for these options will not overwrite the config file values. Instead, all values will be composed together and set to the option.

  ``` text
  [<general-mode>]
  <parameter>=<value1>,<value2>,...
  ```

### 4.3.9. Kernel Renaming

In some cases, it gets difficult to distinguish between results using the kernel function or mangled name. However, demangled names can be quite long and complex to understand. To handle such cases, kernel demangled names are auto-simplified to some extent. To see original kernel demangled names, disable kernel renaming using `--rename-kernels`` ``off` option. If a simplified kernel demangled name turns out to be not useful, you can rename it through the configuration file. A kernel renaming configuration file should be a YAML file written in the following format:

``` text
-
 - Original: mergeRanksAndIndicesKernel(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int)
 - Renamed: Merge Rank Kernel
-
 - Original: void mergeSortSharedKernel<(unsigned int)1>(unsigned int *, unsigned int *, unsigned int *, unsigned int *, unsigned int)
 - Renamed: Merge Sort Kernel
```

By default, kernel renaming config file with name `ncu-kernel-renames.yaml` is searched in the similar way as [config file](index.html#config-file) is searched.

To avoid manually writing demangled names in the config file, one can use `--rename-kernels-export`` ``on` option to export demangled names from the report to the config file with mappings for renaming them.

Using the `--rename-kernels-path`` ``<path>` option, default path and name of the file used while importing renamed names and exporting can be overwritten.

Note that renamed names can later be used to filter kernels in the report using `--kernel-name` and `--kernel-id` options.

Similary, you can use the renamed names to filter the kernels while profiling as well.

**Sample usage**

To profile a kernel with demangled name ‘void SomeComplexKernel(unsigned int, unsigned int, unsigned int)’, you can import the following kernel renaming configuration file:

``` text
-
 - Original: void SomeComplexKernel(unsigned int, unsigned int, unsigned int)
 - Renamed: MySimpleKernel
```

``` text
ncu
--rename-kernels-path /path/to/ncu-kernel-renames.yaml
--kernel-name-base demangled
--kernel-name MySimpleKernel
app.exe
```

If you export the report from the above command, you will see the kernel demangled name renamed to ‘MySimpleKernel’ in the report even if you remove the renaming configuration file. If you want to see the original demangled name in the report, you can always rename it again using the renaming configuration file or simply disable kernel renaming using `--rename-kernels`` ``off` option. Note that the renaming can be done on demangled names only. That’s why you need to use `--kernel-name-base`` ``demangled` option to use the renamed demangled names for kernel filtering.

You can also use the kernel renaming configuration file while exporting kernels from one report to another using [Filtered Profile Export](index.html#filtered-profile-export) feature. For example, if you rename the kernels in the configuration file as:

``` text
-
 - Original: KernelA(unsigned int, unsigned int, unsigned int)
 - Renamed: KernelA_Export
-
 - Original: KernelB(unsigned int, unsigned int, unsigned int)
 - Renamed: KernelB_NoExport
-
 - Original: KernelC(unsigned int, unsigned int, unsigned int)
 - Renamed: KernelC_Export
```

And then use this configuration file while exporting kernels with the following command:

``` text
ncu
--export /path/to/NewReport.ncu-rep
--import /path/to/OldReport.ncu-rep
--rename-kernels-path /path/to/ncu-kernel-renames.yaml
--kernel-name-base demangled
--kernel-name regex:.*_Export
```

Then, only the kernels with renamed names ‘KernelA_Export’ and ‘KernelC_Export’ will be exported to the new report.

### 4.3.10. CPU Call Stack Filtering

CPU call stack filtering allows you to filter the launches based on their CPU call stacktrace. You can collect two types of CPU call stacks: “Native” and “Python” using the `--call-stack` or `--call-stack-type` option. You can filter the launches based on the collected call stack using the following options:

`--native-include`` ``<configuration>`` ``--native-exclude`` ``<configuration>`

Through these options, you can choose kernel launches via a specific native call stack frame function or set of functions to include or exclude from the profiling.

`--python-include`` ``<configuration>`` ``--python-exclude`` ``<configuration>`

Through these options, you can choose kernel launches via a specific python call stack frame function or set of functions to include or exclude from the profiling.

These options are used to profile only those kernels which satisfy the conditions mentioned in the configurations. You can use each option multiple times, mentioning all the “include” and “exclude” configurations as needed. In order to use these include/exclude options, you must collect the respective call stack type using the `--call-stack` or `--call-stack-type` option. However, if you have already collected the call stack into the report file, you can directly use the include/exclude options to filter the results while importing the report.

Include/Exclude configuration follows the below format:

`<Module>@<File>@<Function>`

where \<Module\> and \<File\> are optional. If you do not provide any one of them, your configuration will be searched in all the modules or files respectively. You must provide the \<Function\> expression otherwise the configuration will be considered invalid.

- \<File\>@\<Function\> usage is similar to the usage of \<Domain\>@\<Range\> in [NVTX Filtering](index.html#nvtx-filtering)

  > [TABLE]
  >
  > Include kernels launched via function ‘FuncA’ in file ‘FileA.py’ across all modules:
  >
  > ``` text
  > ncu --call-stack-type python --python-include FileA.py@FuncA python CuPythonScript.py
  > ```
  >
  > Include kernels launched via function ‘FuncA’ in file ‘FileA.cpp’ and ‘ModuleA:
  >
  > ``` text
  > ncu --call-stack-type native --native-include ModuleA@FileA.cpp@FuncA CuApp.exe
  > ```
  >
  > Exclude kernels launched via function ‘FuncA’ of a native call stack frame but include kernels launched via function ‘FuncB’ of a python call stack frame:
  >
  > ``` text
  > ncu --call-stack-type native --call-stack-type python --native-exclude FuncA --python-include FuncB python CuPythonScript.py
  > ```

- **Regular Expression Support**

  To use regular expressions, follow the following syntax.

  - Provide prefix ‘regex:’ to treat the config as regular expression.

    The kernels launched via function with matching regex ‘Func\[0-9\]’ of file with matching regex ‘File\[A-Z\]’ are profiled.

    ``` text
    ncu --call-stack-type native --native-include "regex:File[A-Z]@Func[0-9]" CuApp.exe
    ```

    The kernels launched via function with matching regex ‘Func\[0-9\]’ of file with matching regex ‘File\[A-Z\]’ of module with matching regex ‘Module\[A-Z\]’ are profiled.

    ``` text
    ncu --call-stack-type native --native-include "regex:Module[A-Z]@File[A-Z]@Func[0-9]" CuApp.exe
    ```

- **Additional Information**

  ``` text
  ncu --call-stack-type native --native-include "FileA.cpp@FuncA,FileB.cpp@FuncB" CuApp.exe
  ```

  This will not work as you might expect. You cannot pass multiple configurations in a single include/exclude option. You must use the option multiple times to pass multiple configurations.

  ``` text
  ncu --call-stack-type native --native-include @FuncA --native-include @FileA.cpp@FuncB CuApp.exe
  ```

  Both configurations are invalid as you cannot pass an empty \<file\> or \<module\> before the delimiter ‘@’.

Look at [NVTX Push/Pop Range Filtering](index.html#nvtx-filtering) section to understand the function expression format in detail.

## 4.4. Command Line Options

For long command line options, passing a unique initial substring can be sufficient.

### 4.4.1. General

[TABLE]

General Command Line Options {#id22 .table-no-stripes .docutils .align-default}

### 4.4.2. Launch

[TABLE]

Launch Command Line Options {#id23 .table-no-stripes .docutils .align-default}

### 4.4.3. Attach

| Option | Description | Default |
|----|----|----|
| hostname | Set the hostname or IP address for connecting to the machine on which the target application is running. When attaching to a local target application, use 127.0.0.1. | 127.0.0.1 |

Attach Command Line Options {#id24 .table-no-stripes .docutils .align-default}

### 4.4.4. Profile

[TABLE]

Profile Command Line Options {#id25 .table-no-stripes .docutils .align-default}

1([1](#id7),[2](#id8),[3](#id9),[4](#id10),[5](#id11),[6](#id12),[7](#id13),[8](#id14),[9](#id15),[10](#id16),[11](#id17),[12](#id18),[13](#id19),[14](#id20),[15](#id21))  
This filtering option is available when using `--import`.

### 4.4.5. PM Sampling

These options apply to [PM sampling](../ProfilingGuide/index.html#pm-sampling). See [here](index.html#warp-sampling) for options used in warp state sampling.

| Option | Description | Default |  |
|----|----|----|----|
| pm-sampling-interval | Set the PM sampling interval in cycles or ns (depending on the architecture), or determine dynamically when 0. | 0 (auto) |  |
| pm-sampling-buffer-size - | Set the size of the device-sided allocation for PM sampling in bytes, or determine dynamically when 0. | 0 (auto) |  |
| pm-sampling-max-passes - | Set the maximum number of passes used for PM sampling, or determine dynamically when 0. | 0 (auto) |  |
| warp-samples-per-interval | Set the number of warp samples per PM sampling interval, or determine dynamically when 0. | 0 (auto) |  |
| disable-pm-warp-sampling | Disables warp state sampling during PM sampling. | By default, warp state sampling is enabled with the PM sampling. |  |

PM Sampling Command Line Options {#id26 .table-no-stripes .docutils .align-default}

### 4.4.6. Warp Sampling

These options apply to [Warp State Sampling](../ProfilingGuide/index.html#statistical-sampler). See [here](index.html#pm-sampling) for options used in PM sampling.

| Option | Description | Default |
|----|----|----|
| warp-sampling-interval | Set the sampling period in the range of \[0..31\]. The actual frequency is 2 ^ (5 + value) cycles. If set to ‘auto’, the profiler tries to automatically determine a high sampling frequency without skipping samples or overflowing the output buffer. | auto |
| warp-sampling-max-passes | Set maximum number of passes used for sampling (see the [Profiling Guide](../ProfilingGuide/index.html#overhead) for more details on profiling overhead). | 5 |
| warp-sampling-buffer-size | Set the size of the device-sided allocation for samples in bytes. | 32\*1024\*1024 |

Warp Sampling Command Line Options {#id27 .table-no-stripes .docutils .align-default}

### 4.4.7. File

| Option | Description | Default |
|----|----|----|
| log-file | Send all tool output to the specified file, or one of the standard channels. The file will be overwritten. If the file doesn’t exist, a new one will be created.”stdout” as the whole file name indicates standard output channel (stdout). “stderr” as the whole file name indicates standard error channel (stderr).” | If `--log-file` is not set , profile results will be printed on the console. |
| o,export | Set the output file for writing the profile report. If not set, a temporary file will be used which is removed afterwards. Use with `--import` option to save filtered results. See [Filtered Profile Export](index.html#filtered-profile-export) for more details. The specified name supports macro expansion. See [File Macros](index.html#command-line-options-file-macros) for more details. | If `--export` is set and no `--page` option is given, no profile results will be printed on the console. |
| f,force-overwrite | Force overwriting all output files. | By default, the profiler won’t overwrite existing output files and show an error instead. |
| i,import | Set the input file for reading the profile results. |  |
| open-in-ui | Open report in UI instead of showing result on terminal. (Only available on host platforms) |  |
| section-folder-restore | Restores stock files to the default section folder or the folder specified by an accompanying *–section-folder* option. If the operation will overwrite modified files then the *–force-overwrite* option is required. |  |

File Command Line Options {#id28 .table-no-stripes .docutils .align-default}

### 4.4.8. Console Output

[TABLE]

Console Output Command Line Options {#id29 .table-no-stripes .docutils .align-default}

### 4.4.9. Response File

Response files can be specified by adding `@FileName` to the command line. The file name must immediately follow the `@` character. The content of each response file is inserted in place of the corresponding response file option.

### 4.4.10. File Macros

The file name specified with option `-o` or `--export` supports the following macro expansions. Occurrences of these macros in the report file name are replaced by the corresponding character sequence. If not specified otherwise, the macros cannot be used as part of the file path.

| Macro | Description |
|----|----|
| %h | Expands to the host name of the machine on which the command line profiler is running. |
| %q{ENV_NAME} | Expands to the content of the variable with the given name `ENV_NAME` from the environment of the command line profiler. |
| %p | Expands to the process ID of the command line profiler. |
| %i | Expands to the lowest unused positive integer number that guarantees the resulting file name is not yet used. This macro can only be used once in the output file name. |
| %% | Expands to a single `%` character in the output file name. This macro can be used in the file path and the file name. |

Macro Expansions {#id30 .table-no-stripes .docutils .align-default}

### 4.4.11. MPS

[TABLE]

MPS Command Line Options {#id31 .table-no-stripes .docutils .align-default}

| mps-timeout | mps-num-clients | Behavior |
|----|----|----|
| Specified | Not Specified | Control Process waits for the specified timeout and if after the timeout no clients are discovered, it exits. |
| Not Specified | Specified | Control process waits until the specified number of clients are discovered. |
| Specified | Specified | Control Process waits up to the specified timeout and if number of clients discovered are not at least the number of clients specified, it exits. |
| Not Specified | Not Specified | **Error**: At least one of `--mps-timeout` or `--mps-num-clients` must be specified. |

Behavior for Combinations of `mps-timeout` and `mps-num-clients` {#id32 .table-no-stripes .docutils .align-default}

## 4.5. Environment Variables

The following environment variables can be set before launching NVIDIA Nsight Compute CLI, or the UI, respectively.

[TABLE]

Environment Variables {#id33 .table-no-stripes .docutils .align-default}

## 4.6. Nvprof Transition Guide

Refer to the [older documentation](../Archives/index.html).

Notices

Notices

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

Information furnished is believed to be accurate and reliable. However, NVIDIA Corporation assumes no responsibility for the consequences of use of such information or for any infringement of patents or other rights of third parties that may result from its use. No license is granted by implication of otherwise under any patent rights of NVIDIA Corporation. Specifications mentioned in this publication are subject to change without notice. This publication supersedes and replaces all other information previously supplied. NVIDIA Corporation products are not authorized as critical components in life support devices or systems without express written approval of NVIDIA Corporation.

Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.


---


<!-- source: https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html -->
# 1. Customization Guide

Nsight Compute Customization Guide.

User manual on customizing NVIDIA Nsight Compute tools or integrating them with custom workflows. Information on writing section files, rules for automatic result analysis and scripting access to report files.

## 1.1. Introduction

The goal of NVIDIA Nsight Compute is to design a profiling tool that can be easily extended and customized by expert users. While we provide useful defaults, this allows adapting the reports to a specific use case or to design new ways to investigate collected data. All the following is data driven and does not require the tools to be recompiled.

While working with section files or rules files it is recommended to open the *Metric Selection* tool window from the *Profile* menu item. This tool window lists all sections and rules that were loaded. Rules are grouped as children of their associated section or grouped in the *\[Independent Rules\]* entry. For files that failed to load, the table shows the error message. Use the *Reload* button to reload rule files from disk.

## 1.2. Metric Sections

The [Details](../NsightCompute/index.html#profiler-report-details-page) page consists of [sections](../ProfilingGuide/index.html#sets-and-sections) that focus on a specific part of the kernel analysis each. Every section is defined by a corresponding [section file](index.html#section-files) that specifies the data to be collected as well as the visualization used in the UI or CLI output for this data. Simply modify a deployed section file to add or modify what is collected.

### 1.2.1. Section Files

The section files delivered with the tool are stored in the `sections` sub-folder of the NVIDIA Nsight Compute install directory. Each section is defined in a separate file with the `.section` file extension. At runtime, the installed stock sections (and rules) are deployed to a user-writable directory. This can be disabled with an [environment variable](../NsightComputeCli/index.html#environment-variables). Section files from the deployment directory are loaded automatically at the time the UI connects to a target application or the command line profiler is launched. This way, any changes to section files become immediately available in the next profile run.

A section file is a text representation of a *Google Protocol Buffer* message. The full definition of all available fields of a section message is given in [Section Definition](index.html#section-definition). In short, each section consists of a unique *Identifier* (no spaces allowed), a *Display Name*, an optional *Order* value (for sorting the sections in the [Details page](../NsightCompute/index.html#profiler-report-details-page)), an optional *Description* providing guidance to the user, an optional header table, an optional list of metrics to be collected but not displayed, optional bodies with additional UI elements, and other elements. See `ProfilerSection.proto` for the exact list of available elements. A small example of a very simple section is:

``` text
Identifier: "SampleSection"
DisplayName: "Sample Section"
Description: "This sample section shows information on active warps and cycles."
Header {
  Metrics {
    Label: "Active Warps"
    Name: "smsp__active_warps_avg"
  }
  Metrics {
    Label: "Active Cycles"
    Name: "smsp__active_cycles_avg"
  }
}
```

On data collection, this section will cause the two PerfWorks metrics `smsp__active_warps_avg` and `smsp__active_cycles_avg` to be collected.



The section as shown on the Details page

By default, when not available, metrics specified in section files will only generate a warning during data collection, and would then show up as “N/A” in the UI or CLI. This is in contrast to metrics requested via `--metrics` which would cause an error when not available. How to specify metrics as required for data collection is described in [Metric Options and Filters](index.html#metric-options).

More advanced elements can be used in the body of a section. See the `ProfilerSection.proto` file for which elements are available. The following example shows how to use these in a slightly more complex example. The usage of regexes is allowed in tables and charts in the section *Body* only and follows the format `regex:` followed by the actual regex to match *PerfWorks* metric names.

The supported list of metrics that can be used in sections can be queried using the [command line interface](../NsightComputeCli/index.html#command-line-options-profile) with the `--query-metrics` option. Each of these metrics can be used in any section and will be automatically collected if they appear in any enabled section. Note that even if a metric is used in multiple sections, it will only be collected once. Look at all the shipped sections to see how they are implemented.

``` text
Identifier: "SampleSection"
DisplayName: "Sample Section"
Description: "This sample section shows various metrics."
Header {
  Metrics {
    Label: "Active Warps"
    Name: "smsp__active_warps_avg"
  }
  Metrics {
    Label: "Active Cycles"
    Name: "smsp__active_cycles_avg"
  }
}
Body {
  Items {
    Table {
      Label: "Example Table"
      Rows: 2
      Columns: 1
      Metrics {
        Label: "Avg. Issued Instructions Per Scheduler"
        Name: "smsp__inst_issued_avg"
      }
      Metrics {
        Label: "Avg. Executed Instructions Per Scheduler"
        Name: "smsp__inst_executed_avg"
      }
    }
  }
  Items {
    Table {
      Label: "Metrics Table"
      Columns: 2
      Order: ColumnMajor
      Metrics {
        Name: "regex:.*__elapsed_cycles_sum"
      }
    }
  }
  Items {
    BarChart {
      Label: "Metrics Chart"
      CategoryAxis {
        Label: "Units"
      }
      ValueAxis {
        Label: "Cycles"
      }
      Metrics {
        Name: "regex:.*__elapsed_cycles_sum"
      }
    }
  }
}
```



The output of this section would look similar to this screenshot in the UI

### 1.2.2. Section Definition

Protocol buffer definitions are in the NVIDIA Nsight Compute installation directory under `extras/FileFormat`. To understand section files, start with the definitions and documentation in `ProfilerSection.proto`.

To see the list of available *PerfWorks* metrics for any device or chip, use the `--query-metrics` option of the [command line](../NsightComputeCli/index.html#command-line-options-profile).

### 1.2.3. Metric Options and Filters

Sections allow the user to specify alternative *options* for metrics that have a different metric name on different GPU architectures. Metric options use a min-arch/max-arch range *filter*, replacing the base metric with the first metric option for which the current GPU architecture matches the filter. While not strictly enforced, options for a base metric are expected to share the same meaning and subsequently unit, etc., with the base metric.

In addition to its options, the base metric can be filtered by the same criteria. This is useful for metrics that are only available for certain architectures or in limited collection scopes. See `ProfilerMetricOptions.proto` for which filter options are available.

In the below example, the metric `dram__cycles_elapsed.avg.per_second` is collected on SM 7.0 and SM 7.5-8.6, but not on any in between. It uses the same metric name on these architectures.

``` text
Metrics {
    Label: "DRAM Frequency"
    Name: "dram__cycles_elapsed.avg.per_second"
    Filter {
      MaxArch: CC_70
    }
    Options {
      Name: "dram__cycles_elapsed.avg.per_second"
      Filter {
        MinArch: CC_75
        MaxArch: CC_86
      }
    }
}
```

In the next example, the metric in the section header is only collected for launch-based collection scopes (i.e. kernel- and application replay for CUDA kernels or CUDA Graph nodes), but not in range-based scopes.

``` text
Header {
  Metrics {
    Label: "Theoretical Occupancy"
    Name: "sm__maximum_warps_per_active_cycle_pct"
    Filter {
      CollectionFilter {
        CollectionScopes: CollectionScope_Launch
      }
    }
  }
}
```

Similarly, `CollectionFilter`s can be used to set the `Importance` of a metric, which specifies an expectation on its availability during data collection. `Required` metrics, for instance, are expected to be collectable and would generate an error in case they are not available, whereas `Optional` metrics would only generate a warning. Here is a minimal example, illustrating the functionality:

``` text
Metrics {
  Label: "Compute (SM) Throughput"
  Name: "sm__throughput.avg.pct_of_peak_sustained_elapsed"
  Filter {
    CollectionFilter {
      Importance: Required
    }
  }
}
```

Filters can be applied to an entire section instead of or in addition to being set for individual metrics. If both types of filters are specified, they are combined, such that `Metrics`-scope filters take precedence over section-scope filters.

### 1.2.4. Custom Descriptions

Section files support to specify custom descriptions in many places where metrics can be used. Specifying custom descriptions should only be required when creating [derived metrics](index.html#derived-metrics) or when adding new metrics through the [rule system](index.html#rule-system). When setting this for a metric that would otherwise have a description, the existing one is overriden by the custom value.

``` text
Metrics {
  Label: "Custom Metric"
  Name: "custom_metric"
  Description: "Metric added when the rule associated with this section file is triggered."
}
```

Note that the `Description` field is currently only supported for the section `Metrics` field, but not for individual [Options](index.html#metric-options-and-filters).

### 1.2.5. PM Sampling Timelines

PM sampling metrics in section files should be specified in timeline fields. Timelines are items of section Body fields. Timeslines are composed of any number of MetricGroups with any number of MetricRows, each. MetricGroups can be expanded or collapsed in the timeline UI. Each MetricRow can contain any number of Metrics. If a row contains multiple metrics, their values are aggregated.

Each timeline metric should be associated with a Groups field. All metrics in the same group are collected in the same replay pass. Metrics can have optional Multiplier fields. The value of the metric is multiplied with it before being recorded in the report.

``` text
Timeline {
  MetricGroups {
    Label: "Overview"
    Expanded: true
    MetricRows {
      Metrics {
        Label: "Average Active Warps Per Cycle"
        Name: "TriageAC.tpc__warps_active_realtime.avg.per_cycle_active"
        Groups: "sampling_0"
        Multiplier: 0.5
      }
    }
    MetricRows {
      Metrics {
        Label: "Total Active Warps Per Cycle"
        Name: "TriageAC.tpc__warps_active_realtime.sum.per_cycle_active"
        Groups: "sampling_0"
      }
    }
  }

  MetricGroups {
    Label: "SM"
    Expanded: true
    MetricRows {
      Metrics {
        Label: "SM Throughput"
        Name: "TriageSCG.sm__throughput.avg.pct_of_peak_sustained_elapsed"
        Groups: "sampling_1"
      }
  }
```

### 1.2.6. Counter Domains

PM sampling metrics are composed of one or more raw counter dependencies internally. Each counter is associated with a [counter domain](../ProfilingGuide/index.html#pm-sampling), which describes how and where in the hardware the counter is collected. For metrics specified in section files, the automatic domain selection can be overwritten when needed to form more optimal PM sampling metric groups.

``` text
Metrics {
  Label: "Short Scoreboard"
  Name: "pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg"
  Groups: "sampling_ws4"
  CtrDomains: "gpu_sm_c"
}
```

Note that the `CtrDomains` field is currently only supported for the section `Metrics` field, but not for individual [Options](index.html#metric-options-and-filters).

### 1.2.7. Missing Sections

If new or updated section files are not used by NVIDIA Nsight Compute, it is most commonly one of two reasons:

**The file is not found:** Section files must have the `.section` extension. They must also be on the section search path. The default search path is the `sections` directory within the installation directory. In NVIDIA Nsight Compute CLI, the search paths can be overwritten using the `--section-folder` and `--section-folder-recursive` options. In NVIDIA Nsight Compute, the search path can be configured in the *Profile* options.

**Syntax errors:** If the file is found but has syntax errors, it will not be available for metric collection. However, error messages are reported for easier debugging. In NVIDIA Nsight Compute CLI, use the `--list-sections` option to get a list of error messages, if any. In NVIDIA Nsight Compute, error messages are reported in the *Metric Selection* tool window.

### 1.2.8. Derived Metrics

Derived Metrics allow you to define new metrics composed of constants or existing metrics directly in a section file. The new metrics are computed at collection time and added permanently to the profile result in the report. They can then subsequently be used for any tables, charts, rules, etc.

NVIDIA Nsight Compute currently supports the following syntax for defining derived metrics in section files:

``` text
MetricDefinitions {
  MetricDefinitions {
    Name: "derived_metric_name"
    Expression: "derived_metric_expr"
  }
  MetricDefinitions {
    ...
  }
  ...
}
```

The actual metric expression is defined as follows:

``` text
derived_metric_expr ::= operand operator operand
operator            ::= + | - | * | /
operand             ::= metric | constant
metric              ::= (an existing metric name)
constant            ::= double | uint64
double              ::= (double-precision number of the form "N.(M)?", e.g. "5." or "0.3109")
uint64              ::= (64-bit unsigned integer number of the form "N", e.g. "2029")
```

Operators are defined as follows:

``` text
For op in (+ | - | *): For each element in a metric it is applied to, the expression left-hand side op-combined with expression right-hand side.
For op in (/): For each element in a metric it is applied to, the expression left-hand side op-combined with expression right-hand side. If the right-hand side operand is of integer-type, and 0, the result is the left-hand side value.
```

Since metrics can contain regular values and/or [instanced values](../ProfilingGuide/index.html#metrics-structure), elements are combined as below. Constants are treated as metrics with only a regular value.

``` text
1. Regular values are operator-combined.
a + b

2. If both metrics have no correlation ids, the first N values are operator-combined, where N is the minimum of the number of elements in both metrics.
a1 + b1
a2 + b2
a3
a4

3. Else if both metrics have correlation ids, the sets of correlation ids from both metrics are joined and then operator-combined as applicable.
a1 + b1
a2
b3
a4 + b4
b5

4. Else if only the left-hand side metric has correlation ids, the right-hand side regular metric value is operator-combined with every element of the left-hand side metric.
a1 + b
a2 + b
a3 + b

5. Else if only the right-hand side metric has correlation ids, the right-hand side element values are operator-combined with the regular metric value of the left-hand side metric.
a + b1 + b2 + b3
```

In all operations, the value kind of the left-hand side operand is used. If the right-hand side operand has a different value kind, it is converted. If the left-hand side operand is a string-kind, it is returned unchanged.

Examples for derived metrics are `derived__avg_thread_executed`, which provides a hint on the number of threads executed on average at each instruction, and `derived__uncoalesced_l2_transactions_global`, which indicates the ratio of actual L2 transactions vs. ideal L2 transactions at each applicable instruction.

``` text
MetricDefinitions {
  MetricDefinitions {
    Name: "derived__avg_thread_executed"
    Expression: "thread_inst_executed_true / inst_executed"
  }
  MetricDefinitions {
    Name: "derived__uncoalesced_l2_transactions_global"
    Expression: "memory_l2_transactions_global / memory_ideal_l2_transactions_global"
  }
  MetricDefinitions {
    Name: "sm__sass_thread_inst_executed_op_ffma_pred_on_x2"
    Expression: "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained * 2"
  }
}
```

## 1.3. Rule System

NVIDIA Nsight Compute features a Python-based rule system.

### 1.3.1. Writing Rules

To create a new rule, you need to create a new text file with the extension `.py` and place it at some location that is detectable by the tool (see Nsight Compute Integration on how to specify the search path for rules). At a minimum, the rule file must implement two functions, `get_identifier` and `apply`. See Rule File API for a description of all functions supported in rule files. See NvRules for details on the interface available in the rule’s `apply` function.

### 1.3.2. Integration

The rule system is integrated into NVIDIA Nsight Compute as part of the profile report view. When you profile a kernel, available rules will be shown in the report’s *Details* page. You can either select to apply all available rules at once by clicking *Apply Rules* at the top of the page, or apply rules individually. Once applied, the rule results will be added to the current report. By default, all rules are applied automatically.



Section with a single Bottleneck rule available.



The same section with the Bottleneck rule applied. It added a single message to the report.



The section Rule has two associated rules, Basic Template Rule and Advanced Template Rule. The latter is not yet applied. Rules can add various UI elements, including warning and error messages as well as charts and tables.



Some rules are applied independently from sections. They are shown under Independent Rules.

### 1.3.3. Rule System Architecture

The rule system consists of the Python interpreter, the *NvRules C++ interface*, the *NvRules Python interface* (NvRules.py) and a set of rule files. Each rule file is valid Python code that imports the NvRules.py module, adheres to certain standards defined by the [Rule File API](index.html#rule-file-api) and is called from the tool.

When applying a rule, a handle to the rule *Context* is provided to its apply function. This context captures most of the functionality that is available to rules as part of the [NvRules API](index.html#nvrules-api). In addition, some functionality is provided directly by the NvRules module, e.g. for global error reporting. Finally, since rules are valid Python code, they can use regular libraries and language functionality that ship with Python as well.

From the rule *Context*, multiple further objects can be accessed, e.g. the [*Frontend*](index.html#ifrontend-api), *Ranges* and *Actions*. It should be noted that those are only interfaces, i.e. the actual implementation can vary from tool to tool that decides to implement this functionality.

Naming of these interfaces is chosen to be as API-independent as possible, i.e. not to imply CUDA-specific semantics. However, since many compute and graphics APIs map to similar concepts, it can easily be mapped to CUDA terminology, too. A *Range* refers to a CUDA stream, an Action refers to a single CUDA kernel instance. Each action references several *Metrics* that have been collected during profiling (e.g. `instructions`` ``executed`) or are statically available (e.g. the launch configuration). *Metrics* are accessed via their names from the *Action*.

Each CUDA stream can contain any number of kernel (or other device activity) instances and so each *Range* can reference one or more *Actions*. However, currently only a single *Action* per *Range* will be available, as only a single CUDA kernel can be profiled at once.

### 1.3.4. NvRules API

The *NvRules API* is defined as a C/C++ style interface, which is converted to the NvRules.py Python module to be consumable by the rules. As such, C++ class interfaces are directly converted to Python classes und functions. See the [NvRules API](../NvRulesAPI/index.html#abstract) documentation for the classes and functions available in this interface.

### 1.3.5. Rule File API

The *Rule File API* is the implicit contract between the rule Python file and the tool. It defines which functions (syntactically and semantically) the Python file must provide to properly work as a rule.

**Mandatory Functions**

- `get_identifier()`: Return the unique rule identifier string.

- `apply(handle)`: Apply this rule to the rule context provided by handle. Use `NvRules.get_context(handle)` to obtain the *Context* interface from handle.

- `get_name()`: Return the user-consumable display name of this rule.

- `get_description()`: Return the user-consumable description of this rule.

**Optional Functions**

- `get_section_identifier()`: Return the unique section identifier that maps this rule to a section. Section-mapped rules will only be available if the corresponding section was collected. They implicitly assume that the metrics requested by the section are collected when the rule is applied.

- `evaluate(handle)`:

  Declare required metrics and rules that are necessary for this rule to be applied. Use `NvRules.require_metrics(handle,`` ``[...])` to declare the list of metrics that must be collected prior to applying this rule.

  Use e.g. `NvRules.require_rules(handle,`` ``[...])` to declare the list of other rules that must be available before applying this rule. Those are the only rules that can be safely proposed by the *Controller* interface.

### 1.3.6. IFrontend API

The *Frontend* provides an interface to manipulate the tool UI by adding messages, graphical elements (such as charts and tables), as well as speedup estimations, focus metrics and source markers.

The most common use case is for a rule to show at least one message, stating the result to the user, as illustrated in `extras/RuleTemplates/BasicRuleTemplate.py`. This could be as simple as “No issues have been detected,” or contain direct hints as to how the user could improve the code, e.g., “Memory is more heavily utilized than Compute. Consider whether it is possible for the kernel to do more compute work.”

The `extras/RuleTemplates` directory contains a number of templates for more advanced use cases of the *Frontend* API, such as adding speedup estimates and focus metrics (`SpeedupWithFocusMetrics.py`) or generic tables (`RuleWithTable.py`) to a rule result.

The full API for the *Frontend* is documented in the [IFrontend API documentation](../NvRulesAPI/index.html#NvRules.IFrontend).

### 1.3.7. Rule Examples

The following example rule determines on which major GPU architecture a kernel was running.

``` python
import NvRules

def get_identifier():
  return "GpuArch"

def apply(handle):
  ctx = NvRules.get_context(handle)
  action = ctx.range_by_idx(0).action_by_idx(0)
  ccMajor = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
  ctx.frontend().message("Running on major compute capability " + str(ccMajor))
```

## 1.4. Source Counters

The *Source* page provides correlation of various metrics with CUDA-C, PTX and SASS source of the application, depending on availability.



Which source metrics are collected and the order in which they are displayed in this page is controlled using section files, specifically using *SourceMetrics* entries. Each *SourceMetrics* entry defines one ordered group of metrics, and can be assigned an optional *Order* value. This value defines the ordering among those groups in the Source page. This allows you to define a group of memory-related source counters in one and a group of instruction-related counters in another section file.

``` text
Identifier: "CustomSourceMetrics"
DisplayName: "Custom Source Metrics"
SourceMetrics {
  Order: 2
  Metrics {
    Label: "Instructions Executed"
    Name: "inst_executed"
  }
  Metrics {
    Label: "Float-value metric"
    Name: "float_value_metric"
    Description: "Custom, optional metric description"
    DisplayProperties {
      SourceView {
        DefaultVisible: false
      }
    }
  }
  Metrics {
    Label: ""
    Name: "collected_but_not_shown"
  }
}
Metrics {
  Order: 3
  Metrics {
    Label: "Uint64-value metric"
    Name: "uint64_value_metric"
}
```

If a *Source Counter* metric is given an empty label attribute in the section file, it will be collected but not shown on the page. Default visibility can also be set through the *DisplayProperties* field as shown in the example above. The *Metrics* groups has been deprecated to define metrics that should be shown on the Source page, and *SourceMetrics* groups should be used instead, as the former only supports metrics with uint64-type instance values.

## 1.5. Report File Format

This section documents the internals of the profiler report files (reports in the following) as created by NVIDIA Nsight Compute.

Note

The file format is subject to change in future releases without prior notice.

Report files can optionally be compressed using [zstd](https://facebook.github.io/zstd/) compression. They will then use the `.ncu-repz` extension instead of the usual `.ncu-rep` extension. When this is the case, the whole format described in this section is encapsulated into a compressed container.

### 1.5.1. Version 7 Format

Reports of version 7 are a combination of raw binary data and serialized Google Protocol Buffer version 2 messages (proto). All binary entries are stored as little endian. Protocol buffer definitions are in the NVIDIA Nsight Compute installation directory under `extras/FileFormat`.

| Offset \[bytes\] | Entry | Type | Value |
|----|----|----|----|
| 0 | Magic Number | Binary | NVR\0 |
| 4 | Integer | Binary | sizeof(File Header) |
| 8 | File Header | Proto | Report version |
| 8 + sizeof(File Header) | Block 0 | Mixed | CUDA CUBIN source, profile results, session information |
| 8 + sizeof(File Header) + sizeof(Block 0) | Block 1 | Mixed | CUDA CUBIN source, profile results, session information |
| … | … | … | … |

Top-level report file format {#id2 .table-no-stripes .docutils .align-default}

| Offset \[bytes\] | Entry | Type | Value |
|----|----|----|----|
| 0 | Integer | Binary | sizeof(Block Header) |
| 4 | Block Header | Proto | Number of entries per payload type, payload size |
| 4 + sizeof(Block Header) | Block Payload | Mixed | Payload (CUDA CUBIN sources, profile results, session information, string table) |

Per-Block report file format {#id3 .table-no-stripes .docutils .align-default}

| Offset \[bytes\] | Entry | Type | Value |
|----|----|----|----|
| 0 | Integer | Binary | sizeof(Payload type 1, entry 1) |
| 4 | Payload type 1, entry 1 | Proto |  |
| 4 + sizeof(Payload type 1, entry 1) | Integer | Binary | sizeof(Payload type 1, entry 2) |
| 8 + sizeof(Payload type 1, entry 1) | Payload type 1, entry 2 | Proto |  |
| … | … | … | … |
| … | Integer | Binary | sizeof(Payload type 2, entry 1) |
| … | Payload type 2, entry 1 | Proto |  |
| … | … | … | … |

Block payload report file format {#id4 .table-no-stripes .docutils .align-default}

Proto files are currently deployed in a flat directory but may need to be arranged in the following directory structure during compilation for their `import` directives to work:  
- CpuStacktrace/CpuStacktrace.proto

- Nvtx/Nvtx.proto

- Nvtx/NvtxCategories.proto

- Profiler/ProfilerMetricOptions.proto

- Profiler/ProfilerResultsCommon.proto

- Profiler/ProfilerStringTable.proto

- ProfilerReport.proto

- ProfilerReport/ProfilerReportCommon.proto

- ProfilerSection/ProfilerSection.proto

- RuleSystem/RuleResults.proto

Notices

Notices

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

Information furnished is believed to be accurate and reliable. However, NVIDIA Corporation assumes no responsibility for the consequences of use of such information or for any infringement of patents or other rights of third parties that may result from its use. No license is granted by implication of otherwise under any patent rights of NVIDIA Corporation. Specifications mentioned in this publication are subject to change without notice. This publication supersedes and replaces all other information previously supplied. NVIDIA Corporation products are not authorized as critical components in life support devices or systems without express written approval of NVIDIA Corporation.

Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.


---


<!-- source: https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html -->
# 2. Python Report Interface

## 2.1. Introduction

NVIDIA Nsight Compute features a Python-based interface to interact with exported report files.

The module is called [`ncu_report`](#module-ncu_report "ncu_report") and works on any Python version from 3.7 [1](#fn1). It can be found in the `extras/python` directory of your NVIDIA Nsight Compute package.

In order to use the Python module, you need a report file generated by NVIDIA Nsight Compute. You can obtain such a file by saving it from the graphical interface or by using the `--export` flag of the command line tool.

The types and functions in the [`ncu_report`](#module-ncu_report "ncu_report") module are a subset of the ones available in the NvRules API. The documentation in this section serves as a tutorial. For a more formal description of the exposed API, please refer to the [NvRules API](../NvRulesAPI/index.html#abstract) documentation.

[1](#id1)  
On Linux machines you will also need a GNU-compatible libc and `libgcc_s.so`.

## 2.2. Basic Usage

In order to be able to import [`ncu_report`](#module-ncu_report "ncu_report") you will either have to navigate to the `extras/python` directory, or add its absolute path to the `PYTHONPATH` environment variable. Then, the module can be imported like any Python module:

``` text
>>> import ncu_report
```

**Importing a report**

Once the module is imported, you can load a report file by calling the [`load_report`](#ncu_report.load_report "ncu_report.load_report") function with the path to the file. This function returns an object of type [`IContext`](#ncu_report.IContext "ncu_report.IContext") which holds all the information concerning that report.

``` text
>>> my_context = ncu_report.load_report("my_report.ncu-rep")
```

**Querying ranges**

When working with the Python module, profiling results are grouped into *ranges* which are represented by [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects. You can inspect the number of *ranges* contained in the loaded report by calling the [`IContext.num_ranges`](#ncu_report.IContext.num_ranges "ncu_report.IContext.num_ranges") member function of an [`IContext`](#ncu_report.IContext "ncu_report.IContext") object and retrieve a *range* by its index using [`IContext.range_by_idx`](#ncu_report.IContext.range_by_idx "ncu_report.IContext.range_by_idx").

``` text
>>> my_context.num_ranges()
1
>>> my_range = my_context.range_by_idx(0)
```

**Querying actions**

Inside a *range*, profiling results are called *actions*. You can query the number of *actions* contained in a given *range* by using the [`IRange.num_actions`](#ncu_report.IRange.num_actions "ncu_report.IRange.num_actions") method of an [`IRange`](#ncu_report.IRange "ncu_report.IRange") object.

``` text
>>> my_range.num_actions()
2
```

In the same way *ranges* can be obtained from an [`IContext`](#ncu_report.IContext "ncu_report.IContext") object by using the [`IContext.range_by_idx`](#ncu_report.IContext.range_by_idx "ncu_report.IContext.range_by_idx") method, individual *actions* can be obtained from [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects by using the [`IRange.action_by_idx`](#ncu_report.IRange.action_by_idx "ncu_report.IRange.action_by_idx") method. The resulting *actions* are represented by the [`IAction`](#ncu_report.IAction "ncu_report.IAction") class.

``` text
>>> my_action = my_range.action_by_idx(0)
```

As mentioned previously, an *action* represents a single profiling result. To query the workloads’s name you can use the [`IAction.name`](#ncu_report.IAction.name "ncu_report.IAction.name") member function of the [`IAction`](#ncu_report.IAction "ncu_report.IAction") class.

``` text
>>> my_action.name()
MyKernel
```

To query the workload type, you can use the [`IAction.workload_type`](#ncu_report.IAction.workload_type "ncu_report.IAction.workload_type") member function of the [`IAction`](#ncu_report.IAction "ncu_report.IAction") class, which will return an integer among the `IAction.WorkloadType_*` values.

For CUDA kernels, you get the following result:

``` text
>>> my_action.workload_type()
0
```

All return values of [`IAction.workload_type`](#ncu_report.IAction.workload_type "ncu_report.IAction.workload_type") are documented in API documentation for the [`IAction`](#ncu_report.IAction "ncu_report.IAction") class.

**Querying metrics**

To get a tuple of all metric names contained within an *action* you can use the [`IAction.metric_names`](#ncu_report.IAction.metric_names "ncu_report.IAction.metric_names") method. It is meant to be combined with the [`IAction.metric_by_name`](#ncu_report.IAction.metric_by_name "ncu_report.IAction.metric_by_name") method which returns an [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") object. However, for the same task you may also use the subscript operator through the [`IAction.__getitem__`](#ncu_report.IAction.__getitem__ "ncu_report.IAction.__getitem__") method, as explained in the [High-Level Interface](index.html#python-report-interface-high-level) section below.

The metric names displayed here are the same as the ones you can use with the `--metrics` flag of NVIDIA Nsight Compute. Once you have extracted a *metric* from an *action*, you can obtain its value by using one of the following three methods:

- [`IMetric.as_string`](#ncu_report.IMetric.as_string "ncu_report.IMetric.as_string") to obtain its value as a Python [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

- [`IMetric.as_uint64`](#ncu_report.IMetric.as_uint64 "ncu_report.IMetric.as_uint64") to obtain its value as a Python [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

- [`IMetric.as_double`](#ncu_report.IMetric.as_double "ncu_report.IMetric.as_double") to obtain its value as a Python [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

For example, to print the display name of the GPU on which the workload was profiled you can query the `device__attribute_display_name` metric.

``` text
>>> display_name_metric = my_action.metric_by_name('device__attribute_display_name')
>>> display_name_metric.as_string()
'NVIDIA GeForce RTX 3060 Ti'
```

Note that accessing a metric with the wrong type can lead to unexpected (conversion) results.

``` text
>>> display_name_metric.as_double()
0.0
```

Therefore, it is advisable to directly use the [High-Level](index.html#python-report-interface-high-level) function [`IMetric.value`](#ncu_report.IMetric.value "ncu_report.IMetric.value"), as explained below.

## 2.3. Python Interoperability

On top of the low-level [NvRules API](../NvRulesAPI/index.html#abstract) the Python Report Interface also implements part of the [Python object model](https://docs.python.org/3/reference/datamodel.html). By implementing special methods, the Python Report Interface’s exposed classes can be used with built-in Python mechanisms such as iteration, string formatting and length querying.

This allows you to access *metrics* objects via the [`IAction.__getitem__`](#ncu_report.IAction.__getitem__ "ncu_report.IAction.__getitem__") method of the [`IAction`](#ncu_report.IAction "ncu_report.IAction") class:

``` text
>>> display_name_metric = my_action["device__attribute_display_name"]
```

There is also a convenience method [`IMetric.value`](#ncu_report.IMetric.value "ncu_report.IMetric.value") which allows you to query the value of a *metric* object without knowledge of its type:

``` text
>>> display_name_metric.value()
'NVIDIA GeForce RTX 3060 Ti'
```

All the available methods of a class, as well as their associated Python docstrings, can be looked up interactively via

``` text
>>> help(ncu_report.IMetric)
```

or similarly for other classes and methods. In your code, you can access the docstrings via the [`__doc__`](https://docs.python.org/3/library/stdtypes.html#definition.__doc__ "(in Python v3.14)") attribute, i.e. `ncu_report.IMetric.value.__doc__`.

## 2.4. Metric attributes

Apart from the possibility to query the [`IMetric.name`](#ncu_report.IMetric.name "ncu_report.IMetric.name") and [`IMetric.value`](#ncu_report.IMetric.value "ncu_report.IMetric.value") of an [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") object, you can also query the following additional metric attributes:

- [`IMetric.metric_type`](#ncu_report.IMetric.metric_type "ncu_report.IMetric.metric_type")

- [`IMetric.metric_subtype`](#ncu_report.IMetric.metric_subtype "ncu_report.IMetric.metric_subtype")

- [`IMetric.rollup_operation`](#ncu_report.IMetric.rollup_operation "ncu_report.IMetric.rollup_operation")

- [`IMetric.unit`](#ncu_report.IMetric.unit "ncu_report.IMetric.unit")

- [`IMetric.description`](#ncu_report.IMetric.description "ncu_report.IMetric.description")

The first method [`IMetric.metric_type`](#ncu_report.IMetric.metric_type "ncu_report.IMetric.metric_type") returns one out of three *enum* values ([`IMetric.MetricType_COUNTER`](#ncu_report.IMetric.MetricType_COUNTER "ncu_report.IMetric.MetricType_COUNTER"), [`IMetric.MetricType_RATIO`](#ncu_report.IMetric.MetricType_RATIO "ncu_report.IMetric.MetricType_RATIO"), [`IMetric.MetricType_THROUGHPUT`](#ncu_report.IMetric.MetricType_THROUGHPUT "ncu_report.IMetric.MetricType_THROUGHPUT")) if the metric is a hardware metric, or [`IMetric.MetricType_OTHER`](#ncu_report.IMetric.MetricType_OTHER "ncu_report.IMetric.MetricType_OTHER") otherwise (e.g. for launch or device attributes).

The method [`IMetric.metric_subtype`](#ncu_report.IMetric.metric_subtype "ncu_report.IMetric.metric_subtype") returns an *enum* value representing the subtype of a metric (e.g. [`IMetric.MetricSubtype_PEAK_SUSTAINED`](#ncu_report.IMetric.MetricSubtype_PEAK_SUSTAINED "ncu_report.IMetric.MetricSubtype_PEAK_SUSTAINED"), [`IMetric.MetricSubtype_PER_CYCLE_ACTIVE`](#ncu_report.IMetric.MetricSubtype_PER_CYCLE_ACTIVE "ncu_report.IMetric.MetricSubtype_PER_CYCLE_ACTIVE")). In case a metric does not have a subtype, [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") is returned. All available values may be found in the documentation for the [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") class, or may be looked up interactively by executing `help(ncu_report.IMetric)`.

[`IMetric.rollup_operation`](#ncu_report.IMetric.rollup_operation "ncu_report.IMetric.rollup_operation") returns the operation which is used to accumulate different values of the same *metric* and can be one of [`IMetric.RollupOperation_AVG`](#ncu_report.IMetric.RollupOperation_AVG "ncu_report.IMetric.RollupOperation_AVG"), [`IMetric.RollupOperation_MAX`](#ncu_report.IMetric.RollupOperation_MAX "ncu_report.IMetric.RollupOperation_MAX"), [`IMetric.RollupOperation_MIN`](#ncu_report.IMetric.RollupOperation_MIN "ncu_report.IMetric.RollupOperation_MIN") or [`IMetric.RollupOperation_SUM`](#ncu_report.IMetric.RollupOperation_SUM "ncu_report.IMetric.RollupOperation_SUM") for averaging, maximum, minimum or summation, respectively. If the *metric* in question does not specify a rollup operation [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") will be returned.

Lastly, [`IMetric.unit`](#ncu_report.IMetric.unit "ncu_report.IMetric.unit") and [`IMetric.description`](#ncu_report.IMetric.description "ncu_report.IMetric.description") return a (possibly empty) string of the metric’s *unit* and a short textual *description* for hardware metrics, respectively.

The above methods can be combined to filter through all *metrics* of a report, given certain criteria:

``` python
for metric in metrics:
    if metric.metric_type() == IMetric.MetricType_COUNTER and \
       metric.metric_subtype() == IMetric.MetricSubtype_PER_SECOND and \
       metric.rollup_operation() == IMetric.RollupOperation_AVG:
        print(f"{metric.name()}: {metric.value()} {metric.unit()}")
```

## 2.5. NVTX Support

The [`ncu_report`](#module-ncu_report "ncu_report") has support for the NVIDIA Tools Extension (NVTX). This comes through the [`INvtxState`](#ncu_report.INvtxState "ncu_report.INvtxState") object which represents the NVTX state of a profiled kernel.

An [`INvtxState`](#ncu_report.INvtxState "ncu_report.INvtxState") object can be obtained from an action by using its [`IAction.nvtx_state`](#ncu_report.IAction.nvtx_state "ncu_report.IAction.nvtx_state") method. It exposes the [`INvtxState.domains`](#ncu_report.INvtxState.domains "ncu_report.INvtxState.domains") method which returns a tuple of integers representing the domains this kernel has state in. These integers can be used with the [`INvtxState.domain_by_id`](#ncu_report.INvtxState.domain_by_id "ncu_report.INvtxState.domain_by_id") method to get an [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") object which represents the state of a domain.

The [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") can be used to obtain a tuple of *Push-Pop*, or *Start-End* ranges using the [`INvtxDomainInfo.push_pop_ranges`](#ncu_report.INvtxDomainInfo.push_pop_ranges "ncu_report.INvtxDomainInfo.push_pop_ranges") and [`INvtxDomainInfo.start_end_ranges`](#ncu_report.INvtxDomainInfo.start_end_ranges "ncu_report.INvtxDomainInfo.start_end_ranges") methods.

There is also a [`IRange.actions_by_nvtx`](#ncu_report.IRange.actions_by_nvtx "ncu_report.IRange.actions_by_nvtx") member function in the [`IRange`](#ncu_report.IRange "ncu_report.IRange") class which allows you to get a tuple of actions matching the NVTX state described in its parameter.

The parameters for the [`IRange.actions_by_nvtx`](#ncu_report.IRange.actions_by_nvtx "ncu_report.IRange.actions_by_nvtx") function are two lists of strings representing the state for which we want to query the actions. The first parameter describes the NVTX states to include while the second one describes the NVTX states to exclude. These strings are in the same format as the ones used with the `--nvtx-include` and `--nvtx-exclude` options.

## 2.6. Sample Script

**NVTX Push-Pop range filtering**

This is a sample script which loads a report and prints the names of all the profiled kernels which were wrapped inside `BottomRange` and `TopRange` *Push-Pop ranges* of the default NVTX domain.

``` python
#!/usr/bin/env python3

import sys

import ncu_report

if len(sys.argv) != 2:
    print("usage: {} report_file".format(sys.argv[0]), file=sys.stderr)
    sys.exit(1)

report = ncu_report.load_report(sys.argv[1])

for range_idx in range(report.num_ranges()):
    current_range = report.range_by_idx(range_idx)
    for action_idx in current_range.actions_by_nvtx(["BottomRange/*/TopRange"], []):
        action = current_range.action_by_idx(action_idx)
        print(action.name())
```

## 2.7. API Reference

This documents the content of the [`ncu_report`](#module-ncu_report "ncu_report") package which can be found in the `extras/python` directory of your NVIDIA Nsight Compute installation.

*class* ncu_report.FocusSeverity  
Enum representing the severity of a focus metric.

DEFAULT  
Default severity.

LOW  
Low severity.

HIGH  
High severity.

&nbsp;

*class* ncu_report.IAction  
The [`IAction`](#ncu_report.IAction "ncu_report.IAction") represents a profile result such as a CUDA kernel in a single range or a range itself in range-based profiling, for which zero or more metrics were collected.

NameBase_DEMANGLED  
Name base for demangled names.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

NameBase_FUNCTION  
Name base for function signature names.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

NameBase_MANGLED  
Name base for mangled names.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

WorkloadType_CMDLIST  
Workload type for CBL command lists.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

WorkloadType_GRAPH  
Workload type for CUDA graphs.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

WorkloadType_KERNEL  
Workload type for CUDA kernels or CUDA graph kernel nodes.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

WorkloadType_RANGE  
Workload type for result ranges.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

\_\_getitem\_\_(*key*)  
Get an [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") object contained in this [`IAction`](#ncu_report.IAction "ncu_report.IAction") by its name.

Parameters  
**key** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The name of the [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") object to retrieve.

Returns  
An [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") object.

Return type  
[`IMetric`](#ncu_report.IMetric "ncu_report.IMetric")

Raises  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – If `key` is not of type [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)").

- [**KeyError**](https://docs.python.org/3/library/exceptions.html#KeyError "(in Python v3.14)") – If `key` is not the name of any [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") object.

\_\_iter\_\_()  
Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the metric names of this [`IAction`](#ncu_report.IAction "ncu_report.IAction").

Returns  
An [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the metric names.

Return type  
[iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

\_\_len\_\_()  
Get the number of [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") objects of this [`IAction`](#ncu_report.IAction "ncu_report.IAction").

Returns  
The number of [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

\_\_str\_\_()  
Get a human-readable representation of this [`IAction`](#ncu_report.IAction "ncu_report.IAction").

Returns  
The name of the kernel the [`IAction`](#ncu_report.IAction "ncu_report.IAction") object represents.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

metric_by_name(*metric_name*)  
Get a single [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") by name.

Parameters  
**metric_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The name of the [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") to retrieve.

Returns  
The [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") object or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if no such metric exists.

Return type  
[`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

metric_names()  
Get the names of all metrics of this [`IAction`](#ncu_report.IAction "ncu_report.IAction").

Returns  
The names of all metrics.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

name(*\*args*)  
Get the name of the result the [`IAction`](#ncu_report.IAction "ncu_report.IAction") object represents.

Parameters  
**name_base** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – The desired name base. Defaults to [`NameBase_FUNCTION`](#ncu_report.IAction.NameBase_FUNCTION "ncu_report.IAction.NameBase_FUNCTION").

Returns  
The name of the result (potentially in a specific name base).

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

nvtx_state()  
Get the NVTX state associated with this action.

Returns  
The associated [`INvtxState`](#ncu_report.INvtxState "ncu_report.INvtxState") or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if no state is associated.

Return type  
[`INvtxState`](#ncu_report.INvtxState "ncu_report.INvtxState") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

ptx_by_pc(*address*)  
Get the PTX associated with an address.

Parameters  
**address** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The address to get PTX for.

Returns  
The PTX associated with the given address. If no PTX is available, the empty string will be returned.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

rule_results()  
Get the tuple of all rules associated with this action.

Returns  
A tuple of rule results.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`IRuleResult`](#ncu_report.IRuleResult "ncu_report.IRuleResult")

rule_results_as_dicts()  
Get the list of all rules data associated with this action.

Returns  
A list of rule data dictionaries. Each rule data dictionary contains the following key-value pairs:  
- ’rule_identifier’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The rule identifier.

- ’name’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The rule name.

- ’section_identifier’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The section identifier.

- ’rule_message’([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")) with the following key-value pairs:  
  - ’title’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The title of rule message.

  - ’message’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The message.

  - ’type’ : ([`MsgType`](#ncu_report.MsgType "ncu_report.MsgType")) The message type.

- ’focus_metrics’([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)"))A list of focus metrics. Each focus metric dictionary contains the following key-value pairs:  
  - ’name’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The name of the focus metric.

  - ’value’ : ([`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) The value of the focus metric.

  - ’severity’ : ([`FocusSeverity`](#ncu_report.FocusSeverity "ncu_report.FocusSeverity")) The severity of the focus metric.

  - ’info’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The information about the focus metric.

- ’speedup_estimation’([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")) with the following key-value pairs:  
  - ’type’ : ([`SpeedupType`](#ncu_report.SpeedupType "ncu_report.SpeedupType")) The speedup type.

  - ’speedup’ : ([`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) The speedup value.

- ’result_tables’([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")) with the following key-value pairs:  
  - ’title’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The title of the result table.

  - ’description’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The description of the result table.

  - ’headers’ : ([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The column headers of the result table.

  - ’data’ : ([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") \[[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") \| [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| `Any` \]) The table data in row-major format. Each column have elements of the same type.

  [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") values support below link formats:  
  - @url:\<hypertext\>:\<external link\>@ - To add a external link for a hypertext.

  - @sass:\<address\>:\<hypertext\>@ - To add a link to the hypertext to open the SASS address line on the Source page.

  - @source:\<file name\>:\<line number\>:\<hypertext\>@ - To add a link to the hypertext to open the source file at the specified line number on the Source page.

  - @section:\<section identifier\>:\<hypertext\>@ - To add a link to the hypertext to jump to the respective section.

Return type  
[`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

sass_by_pc(*address*)  
Get the SASS associated with an address.

Parameters  
**address** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The address to get SASS for.

Returns  
The SASS associated with the given address. If no SASS is available, the empty string will be returned.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

source_files()  
Get the source files associated with this action along with their content.

If content is not available for a file (e.g. because it hadn’t been imported into the report), the file name will map to an empty string.

Returns  
A dictionary mapping source files to their content.

Return type  
[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") to [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

source_info(*address*)  
Get the source info for a function address within this action.

Addresses are commonly obtained as correlation IDs of source-correlated metrics.

Parameters  
**address** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The address to get source info for.

Returns  
The [`ISourceInfo`](#ncu_report.ISourceInfo "ncu_report.ISourceInfo") associated to the given address. If no source info is available, [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") is returned.

Return type  
[`ISourceInfo`](#ncu_report.ISourceInfo "ncu_report.ISourceInfo") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

source_markers()  
Get all the source markers associated with this action.

Returns  
A tuple of source data dictionaries. Each source data [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)") contains the following key-value pairs:  
- ’rule_identifier’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The rule identifier.

- ’section_identifier’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The section identifier.

- ’kind’ : ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) [`MarkerKind`](#ncu_report.MarkerKind "ncu_report.MarkerKind") attribute.

- ’message_type’ : ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) [`MsgType`](#ncu_report.MsgType "ncu_report.MsgType") attribute.

- ’message’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The source marker message.

- ’source_location’([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")) with the following key-value pairs:  
  - ’file_name’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The file name.

  - ’line’ : ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The line number.

- ’source_address’ : ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) Source instruction address. Key available only for kind [`MarkerKind.SASS`](#ncu_report.MarkerKind.SASS "ncu_report.MarkerKind.SASS").

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

timed_warp_samples()  
Get the all the timed warp samples associated with this IAction

Returns  
A list of timed warp sample dictionaries. Each timed warp sample [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)") contains the following key-value pairs:  
- ’timestamp’ : ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The GPU timestamp of the sample in nanoseconds.

- ’pc’ : ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The program counter value of the sample.

- ’stall_reason’ : ([`StallReason`](#ncu_report.StallReason "ncu_report.StallReason")) The stall reason code of the sample.

- ’not_issued’ : ([`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) True if no warp was issued on the SM else False.

Return type  
[`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

workload_type()  
Get the workload type of the action.

Returns  
The workload type.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* ncu_report.IContext  
The [`IContext`](#ncu_report.IContext "ncu_report.IContext") class is the top-level object representing an open report.

It can be created by calling the [`load_report`](#ncu_report.load_report "ncu_report.load_report") function.

\_\_getitem\_\_(*key*)  
Get one or more [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects by index or by slice.

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)")) – The index or slice to retrieve.

Returns: [`IRange`](#ncu_report.IRange "ncu_report.IRange") \| [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`IRange`](#ncu_report.IRange "ncu_report.IRange"): An [`IRange`](#ncu_report.IRange "ncu_report.IRange") object or a [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects.

Raises  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – If `key` is not of type [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") or [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)").

- [**IndexError**](https://docs.python.org/3/library/exceptions.html#IndexError "(in Python v3.14)") – If `key` is out of range for the [`IContext`](#ncu_report.IContext "ncu_report.IContext").

\_\_iter\_\_()  
Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects of this [`IContext`](#ncu_report.IContext "ncu_report.IContext").

Returns  
An [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects.

Return type  
[iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") of [`IRange`](#ncu_report.IRange "ncu_report.IRange")

\_\_len\_\_()  
Get the number of [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects in this [`IContext`](#ncu_report.IContext "ncu_report.IContext").

Returns  
The number of [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

num_ranges()  
Get the number of [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects in this [`IContext`](#ncu_report.IContext "ncu_report.IContext").

Returns  
The number of [`IRange`](#ncu_report.IRange "ncu_report.IRange") objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

range_by_idx(*idx*)  
Get an [`IRange`](#ncu_report.IRange "ncu_report.IRange") object by index.

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The index to retrieve.

Returns  
An [`IRange`](#ncu_report.IRange "ncu_report.IRange") object or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the index is out of range.

Return type  
[`IRange`](#ncu_report.IRange "ncu_report.IRange") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

&nbsp;

*class* ncu_report.IMetric  
Represents a single, named metric. An [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") can carry one value or multiple ones if it is an instanced metric.

MetricType_OTHER  
Metric type for metrics that do not fit in any other category.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricType_COUNTER  
Metric type for counter metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricType_RATIO  
Metric type for ratio metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricType_THROUGHPUT  
Metric type for throughput metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_NONE  
Metric subtype for metrics that do not have a subtype.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED  
Metric subtype for peak sustained metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED_ACTIVE  
Metric subtype for peak sustained active metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED_ACTIVE_PER_SECOND  
Metric subtype for peak sustained active per-second metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED_ELAPSED  
Metric subtype for peak sustained elapsed metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED_ELAPSED_PER_SECOND  
Metric subtype for peak sustained elapsed per-second metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PER_CYCLE_ACTIVE  
Metric subtype for per-cycle active metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PER_CYCLE_ELAPSED  
Metric subtype for per-cycle elapsed metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PER_SECOND  
Metric subtype for per-second metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PCT_OF_PEAK_SUSTAINED_ACTIVE  
Metric subtype for percentage of peak sustained active metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PCT_OF_PEAK_SUSTAINED_ELAPSED  
Metric subtype for percentage of peak sustained elapsed metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_MAX_RATE  
Metric subtype for max rate metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PCT  
Metric subtype for percentage metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_RATIO  
Metric subtype for ratio metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_NONE  
No rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_AVG  
Average rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_MAX  
Maximum rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_MIN  
Minimum rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_SUM  
Sum rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_UNKNOWN  
Unknown value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_ANY  
Undefined value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_STRING  
String value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_FLOAT  
Float value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_DOUBLE  
Double value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_UINT32  
Unsigned 32-bit integer value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_UINT64  
Unsigned 64-bit integer value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

\_\_str\_\_()  
Get a human-readable representation of this [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric").

Returns  
The name of the [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

as_double(*\*args*)  
Get the metric value or metric instance value as a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)").

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance value to retrieve instead a metric value.

Returns  
The metric value or metric instance value requested. If the value cannot be casted to a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"), this function will return `0.`.

Return type  
[`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

as_string(*\*args*)  
Get the metric value or metric instance value as a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)").

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance value to retrieve instead a metric value.

Returns  
The metric value or metric instance value requested. If the value cannot be casted to a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), this function will return [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

as_uint64(*\*args*)  
Get the metric value or metric instance value as an [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)").

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance value to retrieve instead a metric value.

Returns  
The metric value or metric instance value requested. If the value cannot be casted to a [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), this function will return `0`.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

correlation_ids()  
Get a metric object for this metric’s instance value’s correlation IDs.

Returns a new [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") representing the correlation IDs for the metric’s instance values. Use [`IMetric.has_correlation_ids`](#ncu_report.IMetric.has_correlation_ids "ncu_report.IMetric.has_correlation_ids") to check if this metric has correlation IDs for its instance values. Correlation IDs are used to associate instance values with the instance their value represents. In the returned new metric object, the correlation IDs are that object’s instance values.

If the metric does not have any correlation IDs, this function will return [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)").

Returns  
The new [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") object representing the correlation IDs for this metric’s instance values or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the metric has no correlation IDs.

Return type  
[`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

description()  
Get the metric description.

Returns  
The description of the metric.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

has_correlation_ids()  
Check if the metric has correlation IDs.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the metric has correlation IDs matching its instance values, [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

has_value(*\*args*)  
Check if the metric or metric instance has a value.

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance metric to check.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the metric or metric instance has a value, `` False` `` otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

kind(*\*args*)  
Get the metric or metric instance value kind.

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance metric to get the value kind for.

Returns  
The metric or metric instance value kind.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

metric_subtype()  
Get the metric subtype.

Returns  
The metric subtype.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

metric_type()  
Get the metric type.

Returns  
The metric type.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

name()  
Get the metric name.

Returns  
The metric name.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

num_instances()  
Get the number of instance values for this metric.

Not all metrics have instance values. If a metric has instance values, it may also have [`IMetric.correlation_ids`](#ncu_report.IMetric.correlation_ids "ncu_report.IMetric.correlation_ids") matching these instance values.

Returns  
The number of instances for this metric.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

rollup_operation()  
Get the type of rollup operation for this metric.

Returns  
The rollup operation type.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

unit()  
Get the metric unit.

Returns  
The metric unit.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

value(*idx=None*)  
Get the value of this [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric").

This is a convenience function that will wrap the logic of invoking the correct `IMetric.as_*` method based on the value kind of this [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric").

Parameters  
**idx** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – The index of the value to get.

Returns  
The value of this [`IMetric`](#ncu_report.IMetric "ncu_report.IMetric") as [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") or [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"). If no value is available, this will return [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

&nbsp;

*class* ncu_report.INvtxDomainInfo  
Represents a single NVTX domain of the NVTX state, including all ranges associated with this domain.

\_\_str\_\_()  
Get a human-readable representation of this [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo").

Returns  
The name of the [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

name()  
Get a human-readable representation of this [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo").

Returns  
The name of the [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

push_pop_range(*idx*)  
Get a push/pop range object by index.

The index is identical to the range’s order on the call stack.

Returns  
The requested [`INvtxRange`](#ncu_report.INvtxRange "ncu_report.INvtxRange") or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the index is out of range.

Return type  
[`INvtxRange`](#ncu_report.INvtxRange "ncu_report.INvtxRange") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

push_pop_ranges()  
Get a sorted list of push/pop range names.

Get the sorted list of stacked push/pop range names in this domain, associated with the current [`INvtxState`](#ncu_report.INvtxState "ncu_report.INvtxState").

Returns  
The sorted names of all push/pop ranges.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

start_end_range(*idx*)  
Get a start/end range object by index.

Returns  
The requested [`INvtxRange`](#ncu_report.INvtxRange "ncu_report.INvtxRange") or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the index is out of range.

Return type  
[`INvtxRange`](#ncu_report.INvtxRange "ncu_report.INvtxRange") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

start_end_ranges()  
Get a sorted list of start/end range names.

Get the sorted list of start/end range names in this domain, associated with the current [`INvtxState`](#ncu_report.INvtxState "ncu_report.INvtxState").

Returns  
The sorted names of all start/end ranges.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

&nbsp;

*class* ncu_report.INvtxRange  
Represents a single NVTX Push/Pop or Start/End range.

PayloadType_PAYLOAD_UNKNOWN  
Payload type for ranges of unknown type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_INT32  
Payload type for ranges of INT32 type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_INT64  
Payload type for ranges of INT64 type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_UINT32  
Payload type for ranges of UINT32 type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_UINT64  
Payload type for ranges of UINT64 type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_FLOAT  
Payload type for ranges of float type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_DOUBLE  
Payload type for ranges of double type .

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_JSON  
Payload type for ranges of JSON type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

category()  
Get the category attribute value.

Returns  
The category attribute value. If [`INvtxRange.has_attributes`](#ncu_report.INvtxRange.has_attributes "ncu_report.INvtxRange.has_attributes") returns [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)"), this will return `0`.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

color()  
Get the color attribute value.

Returns  
The color attribute value. If [`INvtxRange.has_attributes`](#ncu_report.INvtxRange.has_attributes "ncu_report.INvtxRange.has_attributes") returns [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)"), this will return `0`.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

has_attributes()  
Check if range has event attributes.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the range has event attributes, [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

message()  
Get the message attribute value.

Returns  
The message attribute value. If [`INvtxRange.has_attributes`](#ncu_report.INvtxRange.has_attributes "ncu_report.INvtxRange.has_attributes") returns [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)"), this will return the empty string.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

name()  
Get the range’s name.

Returns  
The range’s name.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

payload_as_double()  
Get the payload attribute value as a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)").

Returns  
The payload attribute’s value as a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"). If the value cannot be casted to a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"), this function will return `0.`.

Return type  
[`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

payload_as_string()  
Get the payload attribute value as a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)").

Returns  
The payload attribute’s value as a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"). If the value cannot be casted to a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), this function will return the empty string.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

payload_as_uint64()  
Get the payload attribute value as an [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)").

Returns  
The payload attribute’s value as a [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"). If the value cannot be casted to a [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), this function will return `0`.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

payload_type()  
Get the payload type as an [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)").

Returns  
The payload type.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* ncu_report.INvtxState  
Represents the NVTX (Nvidia Tools Extensions) state associated with a single [`IAction`](#ncu_report.IAction "ncu_report.IAction").

\_\_getitem\_\_(*key*)  
Get an [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") object by ID.

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The ID of the [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") object.

Returns  
An [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") object.

Return type  
[`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo")

Raises  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – If `key` is not of type [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)").

- [**KeyError**](https://docs.python.org/3/library/exceptions.html#KeyError "(in Python v3.14)") – If `key` is not a valid ID.

\_\_iter\_\_()  
Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") objects of this [`INvtxState`](#ncu_report.INvtxState "ncu_report.INvtxState").

Returns  
An [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") objects.

Return type  
[iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)")

\_\_len\_\_()  
Get the number of [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") objects of this [`INvtxState`](#ncu_report.INvtxState "ncu_report.INvtxState").

Returns  
The number of [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

domain_by_id(*id*)  
Get a [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") object by ID.

Use [`INvtxState.domains`](#ncu_report.INvtxState.domains "ncu_report.INvtxState.domains") to retrieve the list of valid domain IDs.

Parameters  
**id** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The ID of the request domain.

Returns  
The requested [`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") object.

Return type  
[`INvtxDomainInfo`](#ncu_report.INvtxDomainInfo "ncu_report.INvtxDomainInfo") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

domains()  
Get the list of domain IDs in this state.

Returns  
The tuple of valid domain IDs.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* ncu_report.IRange  
Represents a serial, ordered stream of execution, such as a CUDA stream. It holds one or more actions that were logically executing in this range.

\_\_getitem\_\_(*key*)  
Get one or more [`IAction`](#ncu_report.IAction "ncu_report.IAction") objects by index or by [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)").

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)")) – The index or [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)") to retrieve.

Returns  
An [`IAction`](#ncu_report.IAction "ncu_report.IAction") object or a [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`IAction`](#ncu_report.IAction "ncu_report.IAction") objects.

Return type  
[`IAction`](#ncu_report.IAction "ncu_report.IAction") \| [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`IAction`](#ncu_report.IAction "ncu_report.IAction")

Raises  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – If `key` is not of type [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") or [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)").

- [**IndexError**](https://docs.python.org/3/library/exceptions.html#IndexError "(in Python v3.14)") – If `key` is out of range for the [`IRange`](#ncu_report.IRange "ncu_report.IRange").

\_\_iter\_\_()  
Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`IAction`](#ncu_report.IAction "ncu_report.IAction") objects of this class:IRange.

Returns  
An [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`IAction`](#ncu_report.IAction "ncu_report.IAction") objects.

Return type  
[iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") of [`IAction`](#ncu_report.IAction "ncu_report.IAction")

\_\_len\_\_()  
Get the number of [`IAction`](#ncu_report.IAction "ncu_report.IAction") objects in this [`IRange`](#ncu_report.IRange "ncu_report.IRange").

Returns  
The number of class:IAction objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

action_by_idx(*idx*)  
Get an [`IAction`](#ncu_report.IAction "ncu_report.IAction") objects by index.

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The index to retrieve.

Returns  
An [`IAction`](#ncu_report.IAction "ncu_report.IAction") object or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the index is out of range.

Return type  
[`IAction`](#ncu_report.IAction "ncu_report.IAction") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

actions_by_nvtx(*includes*, *excludes*)  
Get a set of indices to IAction objects by their NVTX state. The state is defined using a series of *includes* and *excludes*.

Parameters  
- **includes** ([iterable](https://docs.python.org/3/glossary.html#term-iterable "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The NVTX states the result should be part of.

- **excludes** ([iterable](https://docs.python.org/3/glossary.html#term-iterable "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The NVTX states the result should not be part of.

Returns  
A [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of indices to [`IAction`](#ncu_report.IAction "ncu_report.IAction") matching the desired NVTX state.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

num_actions()  
Get the number of [`IAction`](#ncu_report.IAction "ncu_report.IAction") objects in this [`IRange`](#ncu_report.IRange "ncu_report.IRange").

Returns  
The number of class:IAction objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* ncu_report.IRuleResult  
The [`IRuleResult`](#ncu_report.IRuleResult "ncu_report.IRuleResult") represents rule results.

\_\_str\_\_()  
Get a human-readable name of the rule result.

Returns  
The name of the [`IRuleResult`](#ncu_report.IRuleResult "ncu_report.IRuleResult") object represents.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

focus_metrics()  
Get all the focus metrics details.

Returns  
A list of focus metrics details. Each focus metric dictionary contains the following key-value pairs:  
- ’name’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The name of the focus metric.

- ’value’ : ([`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) The value of the focus metric.

- ’severity’ : ([`FocusSeverity`](#ncu_report.FocusSeverity "ncu_report.FocusSeverity")) The severity of the focus metric.

- ’info’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The information about the focus metric.

Return type  
[`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

has_rule_message()  
Check if the rule has a message.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if rule message present, [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

has_speedup_estimation()  
Check if the rule has speedup estimation.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the rule has speedup estimation, [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

name()  
Get the rule name.

Returns  
The rule name.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

result_tables()  
Get all the result tables.

Returns  
A list of result tables. Each result table dictionary contains the following key-value pairs:  
- ’title’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The title of the result table.

- ’description’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The description of the result table.

- ’headers’ : ([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The column headers of the result table.

- ’data’ : ([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") \[[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") \| [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| `Any` \]) The table data in row-major format. Each column have elements of the same type.

  > class:str values may contain substrings with the following special link formats  
  > - @url:\<hypertext\>:\<external link\>@ - To add a external link for a hypertext.
  >
  > - @sass:\<address\>:\<hypertext\>@ - To add a link to the hypertext to open the SASS address line on the Source page.
  >
  > - @source:\<file name\>:\<line number\>:\<hypertext\>@ - To add a link to the hypertext to open the source file at the specified line number on the Source page.
  >
  > - @section:\<section identifier\>:\<hypertext\>@ - To add a link to the hypertext to jump to the respective section.

Return type  
[`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

rule_identifier()  
Get the rule identifier.

Returns  
The rule identifier.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

rule_message()  
Get the rule message.

Returns  
A dictionary with the following key-value pairs:  
- ’title’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The rule message title.

- ’message’ : ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The rule message.

  > The message may contain substrings with the following special link formats:  
  > - @url:\<hypertext\>:\<external link\>@ - To add a external link for a hypertext.
  >
  > - @sass:\<address\>:\<hypertext\>@ - To add a link to the hypertext to open the SASS address line on the Source page.
  >
  > - @source:\<file name\>:\<line number\>:\<hypertext\>@ - To add a link to the hypertext to open the source file at the specified line number on the Source page.
  >
  > - @section:\<section identifier\>:\<hypertext\>@ - To add a link to the hypertext to jump to the respective section.

- ’type’ : ([`MsgType`](#ncu_report.MsgType "ncu_report.MsgType")) The message type.

Return type  
[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

section_identifier()  
Get the section identifier.

Returns  
The section identifier.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

speedup_estimation()  
Get the speedup estimation.

Returns  
A dictionary with the following key-value pairs  
- ’type’ : ([`SpeedupType`](#ncu_report.SpeedupType "ncu_report.SpeedupType")) The speedup type.

- ’speedup’ : ([`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) The estimated speedup.

Return type  
[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

&nbsp;

*class* ncu_report.ISourceInfo  
Represents the source correlation information for a specific function address within an action.

file_name()  
Get the file name, as embedded in the correlation information.

Returns  
The file name.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

line()  
Get the line number within the file.

Returns  
The line number.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* ncu_report.MarkerKind  
Enum representing the kind of a source marker.

SASS  
The marker will be associated with a SASS instruction.

SOURCE  
The marker will be associated with a Source line.

NONE  
No specific kind of marker.

&nbsp;

*class* ncu_report.MsgType  
Enum representing the type of the message.

NONE  
No specific type for this message.

OK  
The message is informative.

OPTIMIZATION  
The message represents a suggestion for performance optimization.

WARNING  
The message represents a warning or fixable issue.

ERROR  
The message represents an error, potentially in executing the rule.

&nbsp;

*class* ncu_report.SpeedupType  
Enum representing the type of speedup estimation.

UNKNOWN  
Unknown speedup type.

LOCAL  
Value represents increase in hardware efficiency in isolated context.

GLOBAL  
Value represents decrease in overall kernel runtime.

&nbsp;

*class* ncu_report.StallReason  
Enum representing the stall reason code of a timed warp sample.

INVALID  
Invalid stall reason.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

BARRIER  
Warp was stalled waiting for sibling warps at a CTA barrier.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

BRANCH_RESOLVING  
Warp was stalled waiting for a branch to resolve.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

DISPATCH_STALL  
Warp was stalled waiting on a dispatch stall.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

DRAIN  
Warp was stalled waiting after EXIT for all memory instructions to complete so that warp resources can be freed.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

IMC_MISS  
Warp was stalled waiting for an immediate constant cache (IMC) miss.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

LG_THROTTLE  
Warp was stalled waiting for the L1 instruction queue for local and global memory operations to be not full.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

LONG_SCOREBOARD  
Warp was stalled waiting for a scoreboard dependency on an L1TEX (local, global, surface, tex) operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MIO_THROTTLE  
Warp was stalled waiting for the MIO instruction queue to be not full.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MATH_PIPE_THROTTLE  
Warp was stalled waiting for the execution pipe to be available.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MEMBAR  
Warp was stalled on a memory barrier.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MISC  
Warp was stalled for a miscellaneous hardware reason.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

NO_INSTRUCTIONS  
Warp was stalled waiting to be selected to fetch an instruction or waiting on an instruction cache miss.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

NOT_SELECTED  
Warp was stalled waiting for the micro-scheduler to select the warp to issue.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

SELECTED  
Warp was selected by the micro-scheduler and issued an instruction.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

SHORT_SCOREBOARD  
Warp was stalled waiting for a scoreboard dependency on an MIO operation (not to L1TEX).

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

SLEEPING  
Warp was stalled due to all threads in the warp being in the blocked, yielded, or sleep state.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

TEX_THROTTLE  
Warp was stalled waiting for the TEX/L1 instruction queue to be not full.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

WAIT  
Warp was stalled waiting on a fixed latency execution dependency.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

ncu_report.load_report(*file_name*)  
Load an NVIDIA Nsight Compute report file into an [`IContext`](#ncu_report.IContext "ncu_report.IContext") object.

Parameters  
**file_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path "(in Python v3.14)")) – The relative or absolute path to the `.ncu-rep` or `.ncu-repz` report file.

Returns  
An [`IContext`](#ncu_report.IContext "ncu_report.IContext") object representing the loaded report file.

Return type  
[`IContext`](#ncu_report.IContext "ncu_report.IContext")

Raises  
[**FileNotFoundError**](https://docs.python.org/3/library/exceptions.html#FileNotFoundError "(in Python v3.14)") – Either if `file_name` does not exist or if the NVIDIA Nsight Compute library directory cannot be found.

Notices

Notices

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

Information furnished is believed to be accurate and reliable. However, NVIDIA Corporation assumes no responsibility for the consequences of use of such information or for any infringement of patents or other rights of third parties that may result from its use. No license is granted by implication of otherwise under any patent rights of NVIDIA Corporation. Specifications mentioned in this publication are subject to change without notice. This publication supersedes and replaces all other information previously supplied. NVIDIA Corporation products are not authorized as critical components in life support devices or systems without express written approval of NVIDIA Corporation.

Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.


---


<!-- source: https://docs.nvidia.com/nsight-compute/NvRulesAPI/index.html -->
# 3. NvRules API

*class* NvRules.IAction  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

The [`IAction`](#NvRules.IAction "NvRules.IAction") represents a profile result such as a CUDA kernel in a single range or a range itself in range-based profiling, for which zero or more metrics were collected.

NameBase_DEMANGLED  
Name base for demangled names.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

NameBase_FUNCTION  
Name base for function signature names.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

NameBase_MANGLED  
Name base for mangled names.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

WorkloadType_CMDLIST  
Workload type for CBL command lists.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

WorkloadType_GRAPH  
Workload type for CUDA graphs.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

WorkloadType_KERNEL  
Workload type for CUDA kernels or CUDA graph kernel nodes.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

WorkloadType_RANGE  
Workload type for result ranges.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

\_\_getitem\_\_(*key*)  
Get an [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object contained in this [`IAction`](#NvRules.IAction "NvRules.IAction") by its name.

Parameters  
**key** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The name of the [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object to retrieve.

Returns  
An [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object.

Return type  
[`IMetric`](#NvRules.IMetric "NvRules.IMetric")

Raises  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – If `key` is not of type [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)").

- [**KeyError**](https://docs.python.org/3/library/exceptions.html#KeyError "(in Python v3.14)") – If `key` is not the name of any [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object.

\_\_iter\_\_()  
Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the metric names of this [`IAction`](#NvRules.IAction "NvRules.IAction").

Returns  
An [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the metric names.

Return type  
[iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

\_\_len\_\_()  
Get the number of [`IMetric`](#NvRules.IMetric "NvRules.IMetric") objects of this [`IAction`](#NvRules.IAction "NvRules.IAction").

Returns  
The number of [`IMetric`](#NvRules.IMetric "NvRules.IMetric") objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

\_\_str\_\_()  
Get a human-readable representation of this [`IAction`](#NvRules.IAction "NvRules.IAction").

Returns  
The name of the kernel the [`IAction`](#NvRules.IAction "NvRules.IAction") object represents.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

metric_by_name(*metric_name*)  
Get a single [`IMetric`](#NvRules.IMetric "NvRules.IMetric") by name.

Parameters  
**metric_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The name of the [`IMetric`](#NvRules.IMetric "NvRules.IMetric") to retrieve.

Returns  
The [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if no such metric exists.

Return type  
[`IMetric`](#NvRules.IMetric "NvRules.IMetric") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

metric_names()  
Get the names of all metrics of this [`IAction`](#NvRules.IAction "NvRules.IAction").

Returns  
The names of all metrics.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

name(*\*args*)  
Get the name of the result the [`IAction`](#NvRules.IAction "NvRules.IAction") object represents.

Parameters  
**name_base** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – The desired name base. Defaults to [`NameBase_FUNCTION`](#NvRules.IAction.NameBase_FUNCTION "NvRules.IAction.NameBase_FUNCTION").

Returns  
The name of the result (potentially in a specific name base).

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

nvtx_state()  
Get the NVTX state associated with this action.

Returns  
The associated [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState") or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if no state is associated.

Return type  
[`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

ptx_by_pc(*address*)  
Get the PTX associated with an address.

Parameters  
**address** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The address to get PTX for.

Returns  
The PTX associated with the given address. If no PTX is available, the empty string will be returned.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

sass_by_pc(*address*)  
Get the SASS associated with an address.

Parameters  
**address** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The address to get SASS for.

Returns  
The SASS associated with the given address. If no SASS is available, the empty string will be returned.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

source_files()  
Get the source files associated with this action along with their content.

If content is not available for a file (e.g. because it hadn’t been imported into the report), the file name will map to an empty string.

Returns  
A dictionary mapping source files to their content.

Return type  
[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") to [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

source_info(*address*)  
Get the source info for a function address within this action.

Addresses are commonly obtained as correlation IDs of source-correlated metrics.

Parameters  
**address** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The address to get source info for.

Returns  
The `ISourceInfo` associated to the given address. If no source info is available, [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") is returned.

Return type  
`ISourceInfo` \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

workload_type()  
Get the workload type of the action.

Returns  
The workload type.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* NvRules.IContext  
Bases: `NvRules.IBaseContext`

The [`IContext`](#NvRules.IContext "NvRules.IContext") class is the top-level object representing an open report.

It can be created by calling the `load_report` function.

\_\_getitem\_\_(*key*)  
Get one or more [`IRange`](#NvRules.IRange "NvRules.IRange") objects by index or by slice.

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)")) – The index or slice to retrieve.

Returns: [`IRange`](#NvRules.IRange "NvRules.IRange") \| [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`IRange`](#NvRules.IRange "NvRules.IRange"): An [`IRange`](#NvRules.IRange "NvRules.IRange") object or a [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`IRange`](#NvRules.IRange "NvRules.IRange") objects.

Raises  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – If `key` is not of type [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") or [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)").

- [**IndexError**](https://docs.python.org/3/library/exceptions.html#IndexError "(in Python v3.14)") – If `key` is out of range for the [`IContext`](#NvRules.IContext "NvRules.IContext").

\_\_iter\_\_()  
Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`IRange`](#NvRules.IRange "NvRules.IRange") objects of this [`IContext`](#NvRules.IContext "NvRules.IContext").

Returns  
An [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`IRange`](#NvRules.IRange "NvRules.IRange") objects.

Return type  
[iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") of [`IRange`](#NvRules.IRange "NvRules.IRange")

\_\_len\_\_()  
Get the number of [`IRange`](#NvRules.IRange "NvRules.IRange") objects in this [`IContext`](#NvRules.IContext "NvRules.IContext").

Returns  
The number of [`IRange`](#NvRules.IRange "NvRules.IRange") objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

controller()  
Get the controller object.

Returns  
The controller object.

Return type  
[`IController`](#NvRules.IController "NvRules.IController")

frontend()  
Get the frontend object.

Returns  
The frontend object.

Return type  
[`IFrontend`](#NvRules.IFrontend "NvRules.IFrontend")

num_ranges()  
Get the number of [`IRange`](#NvRules.IRange "NvRules.IRange") objects in this [`IContext`](#NvRules.IContext "NvRules.IContext").

Returns  
The number of [`IRange`](#NvRules.IRange "NvRules.IRange") objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

range_by_idx(*idx*)  
Get an [`IRange`](#NvRules.IRange "NvRules.IRange") object by index.

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The index to retrieve.

Returns  
An [`IRange`](#NvRules.IRange "NvRules.IRange") object or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the index is out of range.

Return type  
[`IRange`](#NvRules.IRange "NvRules.IRange") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

&nbsp;

*class* NvRules.IController  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

Controller interface.

The controller can be used to interact with the tool runtime, e.g., to signal the tools to propose a follow-up rule.

get_message_vault()  
Get an [`IMessageVault`](#NvRules.IMessageVault "NvRules.IMessageVault") object that can be used for message passing between rules.

Returns  
[`IMessageVault`](#NvRules.IMessageVault "NvRules.IMessageVault")

propose_rule(*rule*)  
Propose the specified rule in the current context.

Parameters  
**rule** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The proposed rule identifier.

&nbsp;

*class* NvRules.IEvaluator  
Bases: `NvRules.IBaseContext`

Evaluator interface.

The evaluator is used during rule setup to pass information about rule dependencies to the tool. For most cases, its Python wrapper functions [`require_metrics`](#NvRules.require_metrics "NvRules.require_metrics") and [`require_rules`](#NvRules.require_rules "NvRules.require_rules") should be used instead for convenience.

require_metric(*metric*)  
Define that the specified metric must have been collected in order for the calling rule to be applied.

Parameters  
**metric** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The metric name to require.

require_rule(*rule*)  
Define that the specified rule must be available and ready to be applied in order for the calling rule to be applied itself.

Parameters  
**rule** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The rule identifier to require.

&nbsp;

*class* NvRules.IFrontend  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

The frontend is responsible for relaying messages and results to the caller via user interfaces, logs or output files.

Deprecated Attributes and Their Replacement:  
MarkerKind_SASS - [`MarkerKind.SASS`](#NvRules.MarkerKind.SASS "NvRules.MarkerKind.SASS")

MarkerKind_SOURCE - [`MarkerKind.SOURCE`](#NvRules.MarkerKind.SOURCE "NvRules.MarkerKind.SOURCE")

MsgType_MSG_NONE - [`MsgType.NONE`](#NvRules.MsgType.NONE "NvRules.MsgType.NONE")

MsgType_MSG_OK - [`MsgType.OK`](#NvRules.MsgType.OK "NvRules.MsgType.OK")

MsgType_MSG_OPTIMIZATION - [`MsgType.OPTIMIZATION`](#NvRules.MsgType.OPTIMIZATION "NvRules.MsgType.OPTIMIZATION")

MsgType_MSG_WARNING - [`MsgType.WARNING`](#NvRules.MsgType.WARNING "NvRules.MsgType.WARNING")

MsgType_MSG_ERROR - [`MsgType.ERROR`](#NvRules.MsgType.ERROR "NvRules.MsgType.ERROR")

Severity_SEVERITY_DEFAULT  
The default severity.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

Severity_SEVERITY_LOW  
Severity if low.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

Severity_SEVERITY_HIGH  
Severity if high.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

SpeedupType_LOCAL  
The proportional increase in efficiency of the hardware usage when viewing the performance problem in isolation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

SpeedupType_GLOBAL  
The proportional reduction in runtime of the entire workload.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

focus_metric(*message_id*, *metric_name*, *metric_value*, *severity*, *info*)  
Create a rule focus metric message.

Issues a focus metric message to the frontend, e.g. to indicate a key metric that triggered the rule output.

Parameters  
- **message_id** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The prior message to associate with.

- **metric_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Name of the focus metric.

- **metric_value** ([`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – Value of the focus metric.

- **severity** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Indicates the impact orseverity on the result.

- **info** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Descriptive string for further information, e.g., the calculation leading to this metric being focused.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the focus metric could be set successfully,  
[`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

generate_table(*message_id*, *header*, *data*, *config=None*)  
Generate a table in the frontend.

This function attaches a table to the pre-existing rule message given by message_id. It can be called multiple times to attach multiple tables to the same message.

The table can be customized by passing a dict to the config parameter. The table can contain a title, description, a column to sort by, and a global styling. Additionally, the table can have per-column configurations, which can specify a header tooltip, relative column width, and styling for the header and data cells.

This is an example of a valid config dict:

``` python
config = {
    "title": "My table title",
    "description": "Short description of the table",
    "sort_by": {
        "column": "MyColumn",  # specify column by name or index
        "order": "ascending",  # in ascending order (default)
    },
    "per_column_configs": {  # overwrites global styling for individual columns
        "MyColumn": {  # can use column name or index
            "tooltip": "Explanation of MyColumn",  # tooltip for the header
            "relative_width": 0.5,  # relative width of the column
            "style": {
                "header": {"bold": True},
                "data": {"italic": True},
            }
        },
    },
}
```

Parameters  
- **message_id** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The message to which to attach the table to.

- **header** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) – The column labels of the table.

- **data** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *Any\]\]*) –

  The table data in row-major format. Any refers to any type that implements \_\_str\_\_. Each column must only have elements of the same type, and all columns must have the same length.

  class:str values may contain substrings with the following special link formats  
  - @url:\<hypertext\>:\<external link\>@ - To add a external link for a hypertext.

  - @sass:\<address\>:\<hypertext\>@ - To add a link to the hypertext to open the SASS address line on the Source page.

  - @source:\<file name\>:\<line number\>:\<hypertext\>@ - To add a link to the hypertext to open the source file at the specified line number on the Source page.

  - @section:\<section identifier\>:\<hypertext\>@ - To add a link to the hypertext to jump to the respective section.

- **config** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *Any\]* *\|* *None*) – Configuration options for the table. Defaults to None.

Raises  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – If types of elements within a column are mixed.

- [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "(in Python v3.14)") – If columns have different lengths, or len(header) does not match the number of columns in data.

load_chart_from_file(*filename*)  
Load a ProfilerSection google protocol buffer chart from a file.

Parameters  
**filename** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The file name.

message(*\*args*)  
Issues a message to the frontend.

Parameters  
- **type** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – The message type.

- **str** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  The message content.

  The message may contain substrings with the following special link formats:  
  - @url:\<hypertext\>:\<external link\>@ - To add a external link for a hypertext.

  - @sass:\<address\>:\<text\>@ - To add a SASS address to the cell.

  - @source:\<file name\>:\<line number\>:\<hypertext\>@ - To add a link to the hypertext to open the source file at the specified line number on the Source page.

  - @section:\<section identifier\>:\<hypertext\>@ - To add a link to the hypertext to jump to the respective section.

- **name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), optional) – The name of the message.

Returns  
A message ID that is unique in this rule invocation.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

receive_dict_from_parent(*parent_id*)  
Receive a dictionary from a parent rule.

Receive a dictionary of type dict\[str,float\] sent using [`IFrontend.send_dict_to_children`](#NvRules.IFrontend.send_dict_to_children "NvRules.IFrontend.send_dict_to_children"). If the parent id does not represent a pre-specified parent rule of this rule, or in case the parent rule has not been executed, an empty dict will be returned.

Parameters  
**parent_id** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – ID of the pre-specified parent rule.

Returns  
The received dictionary.  
An empty dict if the parent rule has not been executed.

Return type  
[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

send_dict_to_children(*dict*)  
Send a dictionary to all child rules.

Sends a Python dictionary of type dict\[str,float\] to all rules that specify this rule as a parent rule. Child rules can retrieve the message using [`IFrontend.receive_dict_from_parent`](#NvRules.IFrontend.receive_dict_from_parent "NvRules.IFrontend.receive_dict_from_parent"). In case this function is called repeatedly, the dict is updated accordingly, thereby adding new key-value pairs, and overwriting values of pre-existing keys.

Parameters  
**dict** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")) – The dictionary to send.

source_marker(*\*args*)  
Create a rule source marker.

Parameters  
- **str** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The source marker message.

- **address_or_line** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The source location in the appropriate kind of source.

- **kind** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The the kind of source.

- **file_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), optional) – The file name.

- **type** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – The urgency associated with this marker. By default [`MsgType.NONE`](#NvRules.MsgType.NONE "NvRules.MsgType.NONE").

speedup(*message_id*, *type*, *estimated_speedup*)  
Rule estimated speedup message.

Issues an estimated speedup associated with a message to the frontend.

Parameters  
- **message_id** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – ID of the existing message.

- **type** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The SpeedupType. If GLOBAL, it indicates what proportional decrease in workload runtime could potentially be achieved, when following the guidelines of the rule. If LOCAL, it indicates what increase in the efficiency of the hardware usage within the context of the performance problem could be achieved.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the speedup could be set successfully,  
[`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

&nbsp;

*class* NvRules.IMessageVault  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

Passes messages between rules.

Get(*ruleId*)  
Retrieve the message associated with a rule from the vault.

In case the rule is unknown, an empty message is returned.

Parameters  
**ruleId** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The identifier of the rule.

Returns  
The message associated with the rule.

Return type  
[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

Put(*ruleId*, *message*)  
Commit a message associated with a rule to the vault.

In case multiple messages associated with the same rule are committed, the messages are merged, in such a way that new key-value pairs are added, and values of pre-existing keys are updated.

Parameters  
- **ruleId** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The identifier of the rule.

- **message** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")) – The message to commit.

&nbsp;

*class* NvRules.IMetric  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

Represents a single, named metric. An [`IMetric`](#NvRules.IMetric "NvRules.IMetric") can carry one value or multiple ones if it is an instanced metric.

MetricType_OTHER  
Metric type for metrics that do not fit in any other category.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricType_COUNTER  
Metric type for counter metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricType_RATIO  
Metric type for ratio metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricType_THROUGHPUT  
Metric type for throughput metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_NONE  
Metric subtype for metrics that do not have a subtype.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED  
Metric subtype for peak sustained metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED_ACTIVE  
Metric subtype for peak sustained active metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED_ACTIVE_PER_SECOND  
Metric subtype for peak sustained active per-second metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED_ELAPSED  
Metric subtype for peak sustained elapsed metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PEAK_SUSTAINED_ELAPSED_PER_SECOND  
Metric subtype for peak sustained elapsed per-second metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PER_CYCLE_ACTIVE  
Metric subtype for per-cycle active metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PER_CYCLE_ELAPSED  
Metric subtype for per-cycle elapsed metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PER_SECOND  
Metric subtype for per-second metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PCT_OF_PEAK_SUSTAINED_ACTIVE  
Metric subtype for percentage of peak sustained active metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PCT_OF_PEAK_SUSTAINED_ELAPSED  
Metric subtype for percentage of peak sustained elapsed metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_MAX_RATE  
Metric subtype for max rate metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_PCT  
Metric subtype for percentage metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

MetricSubtype_RATIO  
Metric subtype for ratio metrics.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_NONE  
No rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_AVG  
Average rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_MAX  
Maximum rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_MIN  
Minimum rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

RollupOperation_SUM  
Sum rollup operation.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_UNKNOWN  
Unknown value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_ANY  
Undefined value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_STRING  
String value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_FLOAT  
Float value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_DOUBLE  
Double value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_UINT32  
Unsigned 32-bit integer value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

ValueKind_UINT64  
Unsigned 64-bit integer value kind.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

\_\_str\_\_()  
Get a human-readable representation of this [`IMetric`](#NvRules.IMetric "NvRules.IMetric").

Returns  
The name of the [`IMetric`](#NvRules.IMetric "NvRules.IMetric").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

as_double(*\*args*)  
Get the metric value or metric instance value as a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)").

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance value to retrieve instead a metric value.

Returns  
The metric value or metric instance value requested. If the value cannot be casted to a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"), this function will return `0.`.

Return type  
[`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

as_string(*\*args*)  
Get the metric value or metric instance value as a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)").

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance value to retrieve instead a metric value.

Returns  
The metric value or metric instance value requested. If the value cannot be casted to a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), this function will return [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

as_uint64(*\*args*)  
Get the metric value or metric instance value as an [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)").

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance value to retrieve instead a metric value.

Returns  
The metric value or metric instance value requested. If the value cannot be casted to a [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), this function will return `0`.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

correlation_ids()  
Get a metric object for this metric’s instance value’s correlation IDs.

Returns a new [`IMetric`](#NvRules.IMetric "NvRules.IMetric") representing the correlation IDs for the metric’s instance values. Use [`IMetric.has_correlation_ids`](#NvRules.IMetric.has_correlation_ids "NvRules.IMetric.has_correlation_ids") to check if this metric has correlation IDs for its instance values. Correlation IDs are used to associate instance values with the instance their value represents. In the returned new metric object, the correlation IDs are that object’s instance values.

If the metric does not have any correlation IDs, this function will return [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)").

Returns  
The new [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object representing the correlation IDs for this metric’s instance values or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the metric has no correlation IDs.

Return type  
[`IMetric`](#NvRules.IMetric "NvRules.IMetric") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

description()  
Get the metric description.

Returns  
The description of the metric.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

has_correlation_ids()  
Check if the metric has correlation IDs.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the metric has correlation IDs matching its instance values, [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

has_value(*\*args*)  
Check if the metric or metric instance has a value.

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance metric to check.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the metric or metric instance has a value, `` False` `` otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

kind(*\*args*)  
Get the metric or metric instance value kind.

Parameters  
**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – If provided, the index of the instance metric to get the value kind for.

Returns  
The metric or metric instance value kind.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

metric_subtype()  
Get the metric subtype.

Returns  
The metric subtype.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

metric_type()  
Get the metric type.

Returns  
The metric type.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

name()  
Get the metric name.

Returns  
The metric name.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

num_instances()  
Get the number of instance values for this metric.

Not all metrics have instance values. If a metric has instance values, it may also have [`IMetric.correlation_ids`](#NvRules.IMetric.correlation_ids "NvRules.IMetric.correlation_ids") matching these instance values.

Returns  
The number of instances for this metric.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

rollup_operation()  
Get the type of rollup operation for this metric.

Returns  
The rollup operation type.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

unit()  
Get the metric unit.

Returns  
The metric unit.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

value(*idx=None*)  
Get the value of this [`IMetric`](#NvRules.IMetric "NvRules.IMetric").

This is a convenience function that will wrap the logic of invoking the correct `IMetric.as_*` method based on the value kind of this [`IMetric`](#NvRules.IMetric "NvRules.IMetric").

Parameters  
**idx** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – The index of the value to get.

Returns  
The value of this [`IMetric`](#NvRules.IMetric "NvRules.IMetric") as [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") or [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"). If no value is available, this will return [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

&nbsp;

*class* NvRules.IMutableMetric  
Bases: [`NvRules.IMetric`](#NvRules.IMetric "NvRules.IMetric")

Represents a single, named mutable metric. An [`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric") can carry one value or multiple ones if it is an instanced metric. In comparison to [`IMetric`](#NvRules.IMetric "NvRules.IMetric"), [`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric") can be modified by assigning it a new value and/or instance values. The metric kind is determined by the assigned value(s).

mutable_correlation_ids()  
Get a mutable metric object for this metric’s instance value’s correlation IDs.

Returns a new [`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric") representing the correlation IDs for the metric’s instance values. Correlation IDs are used to associate instance values with the instance their value represents. In the returned new metric object, the correlation IDs are that object’s instance values.

Returns  
The new [`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric") object representing the correlation IDs for this metric’s instance values.

Return type  
[`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric")

set_double(*\*args*)  
Assign the metric an instance [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") value.

Parameters  
- **instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – The instance to assign the value to.

- **value_kind** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Acceptable kinds are FLOAT, DOUBLE or ANY (for which the implementation chooses the kind internally)

- **value** ([`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – The value to assign.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the instance value was set, [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

set_string(*\*args*)  
Assign the metric an instance [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") value.

Parameters  
- **instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – The instance to assign the value to.

- **value_kind** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Acceptable kinds are STRING or ANY (for which the implementation chooses the kind internally)

- **value** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The value to assign.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the instance value was set, [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

set_uint64(*\*args*)  
Assign the metric a [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") value.

Parameters  
- **instance** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), optional) – The instance to assign the value to.

- **value_kind** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Acceptable kinds are UINT32, UINT64 or ANY (for which the implementation chooses the kind internally)

- **value** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The value to assign.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the value was set, [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

&nbsp;

*class* NvRules.INvtxDomainInfo  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

Represents a single NVTX domain of the NVTX state, including all ranges associated with this domain.

\_\_str\_\_()  
Get a human-readable representation of this [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo").

Returns  
The name of the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

name()  
Get a human-readable representation of this [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo").

Returns  
The name of the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo").

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

push_pop_range(*idx*)  
Get a push/pop range object by index.

The index is identical to the range’s order on the call stack.

Returns  
The requested [`INvtxRange`](#NvRules.INvtxRange "NvRules.INvtxRange") or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the index is out of range.

Return type  
[`INvtxRange`](#NvRules.INvtxRange "NvRules.INvtxRange") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

push_pop_ranges()  
Get a sorted list of push/pop range names.

Get the sorted list of stacked push/pop range names in this domain, associated with the current [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState").

Returns  
The sorted names of all push/pop ranges.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

start_end_range(*idx*)  
Get a start/end range object by index.

Returns  
The requested [`INvtxRange`](#NvRules.INvtxRange "NvRules.INvtxRange") or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the index is out of range.

Return type  
[`INvtxRange`](#NvRules.INvtxRange "NvRules.INvtxRange") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

start_end_ranges()  
Get a sorted list of start/end range names.

Get the sorted list of start/end range names in this domain, associated with the current [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState").

Returns  
The sorted names of all start/end ranges.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

&nbsp;

*class* NvRules.INvtxRange  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

Represents a single NVTX Push/Pop or Start/End range.

PayloadType_PAYLOAD_UNKNOWN  
Payload type for ranges of unknown type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_INT32  
Payload type for ranges of INT32 type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_INT64  
Payload type for ranges of INT64 type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_UINT32  
Payload type for ranges of UINT32 type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_UINT64  
Payload type for ranges of UINT64 type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_FLOAT  
Payload type for ranges of float type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_DOUBLE  
Payload type for ranges of double type .

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

PayloadType_PAYLOAD_JSON  
Payload type for ranges of JSON type.

Type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

category()  
Get the category attribute value.

Returns  
The category attribute value. If [`INvtxRange.has_attributes`](#NvRules.INvtxRange.has_attributes "NvRules.INvtxRange.has_attributes") returns [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)"), this will return `0`.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

color()  
Get the color attribute value.

Returns  
The color attribute value. If [`INvtxRange.has_attributes`](#NvRules.INvtxRange.has_attributes "NvRules.INvtxRange.has_attributes") returns [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)"), this will return `0`.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

has_attributes()  
Check if range has event attributes.

Returns  
[`True`](https://docs.python.org/3/library/constants.html#True "(in Python v3.14)") if the range has event attributes, [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)") otherwise.

Return type  
[`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

message()  
Get the message attribute value.

Returns  
The message attribute value. If [`INvtxRange.has_attributes`](#NvRules.INvtxRange.has_attributes "NvRules.INvtxRange.has_attributes") returns [`False`](https://docs.python.org/3/library/constants.html#False "(in Python v3.14)"), this will return the empty string.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

name()  
Get the range’s name.

Returns  
The range’s name.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

payload_as_double()  
Get the payload attribute value as a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)").

Returns  
The payload attribute’s value as a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"). If the value cannot be casted to a [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"), this function will return `0.`.

Return type  
[`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

payload_as_string()  
Get the payload attribute value as a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)").

Returns  
The payload attribute’s value as a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"). If the value cannot be casted to a [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), this function will return the empty string.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

payload_as_uint64()  
Get the payload attribute value as an [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)").

Returns  
The payload attribute’s value as a [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"). If the value cannot be casted to a [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), this function will return `0`.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

payload_type()  
Get the payload type as an [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)").

Returns  
The payload type.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* NvRules.INvtxState  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

Represents the NVTX (Nvidia Tools Extensions) state associated with a single [`IAction`](#NvRules.IAction "NvRules.IAction").

\_\_getitem\_\_(*key*)  
Get an [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object by ID.

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The ID of the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object.

Returns  
An [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object.

Return type  
[`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo")

Raises  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – If `key` is not of type [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)").

- [**KeyError**](https://docs.python.org/3/library/exceptions.html#KeyError "(in Python v3.14)") – If `key` is not a valid ID.

\_\_iter\_\_()  
Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") objects of this [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState").

Returns  
An [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") objects.

Return type  
[iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)")

\_\_len\_\_()  
Get the number of [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") objects of this [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState").

Returns  
The number of [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

domain_by_id(*id*)  
Get a [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object by ID.

Use [`INvtxState.domains`](#NvRules.INvtxState.domains "NvRules.INvtxState.domains") to retrieve the list of valid domain IDs.

Parameters  
**id** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The ID of the request domain.

Returns  
The requested [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object.

Return type  
[`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

domains()  
Get the list of domain IDs in this state.

Returns  
The tuple of valid domain IDs.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* NvRules.IRange  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

Represents a serial, ordered stream of execution, such as a CUDA stream. It holds one or more actions that were logically executing in this range.

\_\_getitem\_\_(*key*)  
Get one or more [`IAction`](#NvRules.IAction "NvRules.IAction") objects by index or by [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)").

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)")) – The index or [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)") to retrieve.

Returns  
An [`IAction`](#NvRules.IAction "NvRules.IAction") object or a [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`IAction`](#NvRules.IAction "NvRules.IAction") objects.

Return type  
[`IAction`](#NvRules.IAction "NvRules.IAction") \| [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`IAction`](#NvRules.IAction "NvRules.IAction")

Raises  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – If `key` is not of type [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") or [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)").

- [**IndexError**](https://docs.python.org/3/library/exceptions.html#IndexError "(in Python v3.14)") – If `key` is out of range for the [`IRange`](#NvRules.IRange "NvRules.IRange").

\_\_iter\_\_()  
Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`IAction`](#NvRules.IAction "NvRules.IAction") objects of this class:IRange.

Returns  
An [iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") over the [`IAction`](#NvRules.IAction "NvRules.IAction") objects.

Return type  
[iterator](https://docs.python.org/3/glossary.html#term-iterator "(in Python v3.14)") of [`IAction`](#NvRules.IAction "NvRules.IAction")

\_\_len\_\_()  
Get the number of [`IAction`](#NvRules.IAction "NvRules.IAction") objects in this [`IRange`](#NvRules.IRange "NvRules.IRange").

Returns  
The number of class:IAction objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

action_by_idx(*idx*)  
Get an [`IAction`](#NvRules.IAction "NvRules.IAction") objects by index.

Parameters  
**key** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The index to retrieve.

Returns  
An [`IAction`](#NvRules.IAction "NvRules.IAction") object or [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") if the index is out of range.

Return type  
[`IAction`](#NvRules.IAction "NvRules.IAction") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

actions_by_nvtx(*includes*, *excludes*)  
Get a set of indices to IAction objects by their NVTX state. The state is defined using a series of *includes* and *excludes*.

Parameters  
- **includes** ([iterable](https://docs.python.org/3/glossary.html#term-iterable "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The NVTX states the result should be part of.

- **excludes** ([iterable](https://docs.python.org/3/glossary.html#term-iterable "(in Python v3.14)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The NVTX states the result should not be part of.

Returns  
A [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of indices to [`IAction`](#NvRules.IAction "NvRules.IAction") matching the desired NVTX state.

Return type  
[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

num_actions()  
Get the number of [`IAction`](#NvRules.IAction "NvRules.IAction") objects in this [`IRange`](#NvRules.IRange "NvRules.IRange").

Returns  
The number of class:IAction objects.

Return type  
[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* NvRules.MarkerKind  
Bases: [`enum.IntEnum`](https://docs.python.org/3/library/enum.html#enum.IntEnum "(in Python v3.14)")

Enum representing the kind of a source marker.

SASS  
The marker will be associated with a SASS instruction.

SOURCE  
The marker will be associated with a Source line.

NONE  
No specific kind of marker.

&nbsp;

*class* NvRules.MsgType  
Bases: [`enum.IntEnum`](https://docs.python.org/3/library/enum.html#enum.IntEnum "(in Python v3.14)")

Enum representing the type of the message.

NONE  
No specific type for this message.

OK  
The message is informative.

OPTIMIZATION  
The message represents a suggestion for performance optimization.

WARNING  
The message represents a warning or fixable issue.

ERROR  
The message represents an error, potentially in executing the rule.

&nbsp;

NvRules.get_context(*h*)  
Return the [`IContext`](#NvRules.IContext "NvRules.IContext") object from the context handle.

Parameters  
**h** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The context handle.

Returns  
The evaluator object.

Return type  
[`IContext`](#NvRules.IContext "NvRules.IContext")

&nbsp;

NvRules.get_evaluator(*h*)  
Return the [`IEvaluator`](#NvRules.IEvaluator "NvRules.IEvaluator") object from the context handle.

Parameters  
**h** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The context handle.

Returns  
The evaluator object.

Return type  
[`IEvaluator`](#NvRules.IEvaluator "NvRules.IEvaluator")

&nbsp;

NvRules.get_version(*h*)  
Get version number of this interface.

Parameters  
**h** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The context handle.

Returns  
Returns the version number of this interface as a string of the form \<year\>.\<major\>.\<minor\>. It matches the Nsight Compute version this interface originates from.

Return type  
[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

&nbsp;

NvRules.require_metrics(*handle*, *metrics*)  
Convenience wrapper for [`NvRules.IEvaluator.require_metric`](#NvRules.IEvaluator.require_metric "NvRules.IEvaluator.require_metric").

Parameters  
- **handle** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The context handle, obtained from [`get_context`](#NvRules.get_context "NvRules.get_context").

- **metrics** ([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")) – List of metric names.

&nbsp;

NvRules.require_rules(*handle*, *rules*)  
Convenience wrapper for [`NvRules.IEvaluator.require_rule`](#NvRules.IEvaluator.require_rule "NvRules.IEvaluator.require_rule").

Parameters  
- **handle** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The context handle, obtained from [`get_context`](#NvRules.get_context "NvRules.get_context").

- **rules** ([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")) – List of rule identifiers.

Notices

Notices

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

Information furnished is believed to be accurate and reliable. However, NVIDIA Corporation assumes no responsibility for the consequences of use of such information or for any infringement of patents or other rights of third parties that may result from its use. No license is granted by implication of otherwise under any patent rights of NVIDIA Corporation. Specifications mentioned in this publication are subject to change without notice. This publication supersedes and replaces all other information previously supplied. NVIDIA Corporation products are not authorized as critical components in life support devices or systems without express written approval of NVIDIA Corporation.

Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.


---


<!-- source: https://docs.nvidia.com/nsight-compute/OccupancyCalculatorPythonInterface/index.html -->
# 4. Occupancy Calculator Python Interface

## 4.1. Introduction

NVIDIA Nsight Compute features a Python-based interface for performing occupancy calculations and analysis for kernels on NVIDIA GPUs. These APIs are designed to help developers understand and optimize the utilization of GPU resources to achieve better performance for their kernel.

The module is called [`ncu_occupancy`](#module-ncu_occupancy "ncu_occupancy") and works on any Python version from 3.7 [1](#fn1). It can be found in the `extras/python` directory of your NVIDIA Nsight Compute package.

[1](#id1)  
On Linux machines you will also need a GNU-compatible libc and `libgcc_s.so`.

## 4.2. API Reference

This documents the content of the [`ncu_occupancy`](#module-ncu_occupancy "ncu_occupancy") package which can be found in the `extras/python` directory of your NVIDIA Nsight Compute installation.

*class* ncu_occupancy.OccupancyCalculator  
Provide methods to calculate occupancy and analyze ways to improve it, for a given GPU.

Parameters  
- **computeCapabilityMajor** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The major compute capability version of the GPU.

- **computeCapabilityMinor** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The minor compute capability version of the GPU.

Returns  
An instance of the occupancy calculator.

Return type  
[`OccupancyCalculator`](#ncu_occupancy.OccupancyCalculator "ncu_occupancy.OccupancyCalculator")

get_occupancy_limiters(*occupancy_parameters: [ncu_occupancy.OccupancyParameters](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")*) → [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")  
Get the occupancy limiters for the given occupancy parameters.

Parameters  
**occupancy_parameters** ([`OccupancyParameters`](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")) – The input parameters for the occupancy calculation.

Returns  
[`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`OccupancyLimiter`](#ncu_occupancy.OccupancyLimiter "ncu_occupancy.OccupancyLimiter")

get_optimal_occupancy(*occupancy_parameters: ncu_occupancy.OccupancyParameters, occupancy_variable_list: list = \[\<OccupancyVariable.THREADS_PER_BLOCK: 0\>\]*) → [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")  
Get the optimal occupancy configuration.

Optimal occupancy is calculated by varying input occupancy variable values while keeping other occupancy variable values constant. If no input occupancy variable list provided then [`OccupancyVariable.THREADS_PER_BLOCK`](#ncu_occupancy.OccupancyVariable.THREADS_PER_BLOCK "ncu_occupancy.OccupancyVariable.THREADS_PER_BLOCK") will be considered by default.

Parameters  
- **occupancy_parameters** ([`OccupancyParameters`](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")) – The input parameters for the occupancy calculation.

- **occupancy_variable_list** ([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`OccupancyVariable`](#ncu_occupancy.OccupancyVariable "ncu_occupancy.OccupancyVariable"), optional) – The list of occupancy variables to consider for optimal occupancy calculation. Only up to two occupancy variables can be specified. (default: [`OccupancyVariable.THREADS_PER_BLOCK`](#ncu_occupancy.OccupancyVariable.THREADS_PER_BLOCK "ncu_occupancy.OccupancyVariable.THREADS_PER_BLOCK"))

Returns  
The optimal occupancy configuration. The dictionary contains the following key-value pairs:  
- ’optimal_occupancy’: ([`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) The optimal occupancy.

- ’occupancy_variable_config’:  
  - For single occupancy variable input: [`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of ranges.

  - For two occupancy variables input: [`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") of value combinations. The [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") contains the values in the same order as the input occupancy variable list.

Return type  
[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

get_resource_utilization(*occupancy_parameters: [ncu_occupancy.OccupancyParameters](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")*) → [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")  
Get the resource utilization for the given occupancy parameters.

Parameters  
**occupancy_parameters** ([`OccupancyParameters`](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")) – The input parameters for the occupancy calculation.

Returns  
Resource utilization. The dictionary contains the following key-value pairs:  
- ’sm_occupancy’ : ([`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) The occupancy of the SMs.

- ’allocated_blocks’ : ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The number of allocated blocks out of the total possible blocks per SM.

- ’resource_utilization’([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")) The resource utilization for each resource i.e. threads, registers, shared memory. The resource utilization dictionary contains the following key-value pairs:  
  - ’\<resource name\>’: ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")) The resource utilization for the resource. The resource utilization dictionary contains the following key-value pairs:  
    - ’resource_per_block’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The resource utilized per block.

    - ’unused_resource_count’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The unused resource count per SM.

    - ’unallocated_blocks’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The number of unallocated blocks per SM.

Return type  
[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")

get_sm_occupancy(*occupancy_parameters: [ncu_occupancy.OccupancyParameters](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")*) → [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")  
Calculate the occupancy of the SMs for the given occupancy parameters.

Parameters  
**occupancy_parameters** ([`OccupancyParameters`](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")) – The input parameters for the occupancy calculation.

Returns  
The occupancy of the SMs.

Return type  
[`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

&nbsp;

*class* ncu_occupancy.OccupancyLimiter  
Enum representing the occupancy limiters.

REGISTERS  
Register usage is the occupancy limiter.

SHARED_MEMORY  
Shared memory usage is the occupancy limiter.

BLOCKS  
Block size is the occupancy limiter.

BARRIERS  
Barrier usage is the occupancy limiter.

&nbsp;

*class* ncu_occupancy.OccupancyParameters  
OccupancyParameters is a `dataclass` that holds configuration parameters for occupancy calculations.

shared_mem_size  
Shared memory size configuration (bytes). (default: 0)

Type  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

threads_per_block  
Number of threads per block. (default: 256)

Type  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

registers_per_thread  
Number of registers per thread. (default: 32)

Type  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

shared_mem_per_block  
Shared memory (bytes) per block. (default: 2048)

Type  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

num_block_barriers  
Number of block barriers. (default: 1)

Type  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

&nbsp;

*class* ncu_occupancy.OccupancyVariable  
Enum representing the occupancy variables.

THREADS_PER_BLOCK  
Threads per block.

REGISTERS_PER_THREAD  
Registers per thread.

SHARED_MEMORY_PER_BLOCK  
Shared memory per block.

BLOCK_BARRIERS  
Block barriers.

&nbsp;

ncu_occupancy.get_gpu_data(*major: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*, *minor: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*) → [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")  
Get the GPU data for the given compute capability version.

Parameters  
- **major** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The major compute capability version of the GPU.

- **minor** ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The minor compute capability version of the GPU.

Returns  
The GPU data for the given compute capability version. The dictionary contains the following key-value pairs:  
- ’cc_major’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The major compute capability version of the GPU.

- ’cc_minor’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The minor compute capability version of the GPU.

- ’sm_version’: ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The SM version of the GPU.

- ’chip_family’: ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The chip family of the GPU.

- ’threads_per_warp’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The number of threads per warp.

- ’max_warps_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The number of warps per SM.

- ’max_threads_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The number of threads per SM.

- ’max_thread_blocks_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The number of thread blocks per SM.

- ’block_barriers_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The number of block barriers per SM.

- ’smem_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The shared memory (bytes) per SM.

- ’max_shared_mem_per_block’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The maximum shared memory (bytes) per block.

- ’registers_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The registers per SM.

- ’max_regs_per_block’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The maximum registers per block.

- ’max_regs_per_thread’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The maximum registers per thread.

- ’reg_allocation_unit_size’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The register allocation unit size.

- ’reg_allocation_granularity’: ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The register allocation granularity.

- ’shared_mem_allocation_unit_size’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The shared memory allocation unit size.

- ’warps_allocation_granularity’: ([`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) The warp allocation granularity.

- ’max_thread_block_size’: ([`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The maximum thread block size.

- ’shared_mem_size_configs’: ([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The shared memory size configurations (bytes).

- ’warp_reg_allocation_granularities’: ([`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") of [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) The warp register allocation granularities.

Return type  
[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")
