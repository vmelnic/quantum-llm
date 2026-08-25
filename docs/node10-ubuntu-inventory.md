# Ubuntu node10 inventory

Status: read-only inventory captured on 2026-08-23. No benchmark was run and
no remote configuration was changed.

## Scope and verdict

The private SSH endpoint is:

```bash
ssh vm@192.168.100.11
```

The machine identifies itself as `node10`. It is a Dell PowerEdge R510, not
the previously discussed Tesla P100/P40 rig. As observed, it has no NVIDIA
GPU, CUDA installation or NVMe device. It cannot execute the current Quantum
LLM CUDA providers and is not part of the active self-contained RTX 3090
serving architecture.

This inventory does not establish physical GPU clearance, auxiliary power,
power-supply headroom, cooling capacity or the RAID member layout. Those
properties were not exposed to the unprivileged SSH account and must not be
inferred from the chassis model.

## System summary

| Property | Observed value |
|---|---|
| Hostname | `node10` |
| Vendor and model | Dell PowerEdge R510, hardware revision A02 |
| Motherboard | Dell `0DPRKF`, revision A02 |
| Firmware | Dell BIOS 1.11.0, dated 2012-07-23 |
| Operating system | Ubuntu 26.04 LTS |
| Kernel | `7.0.0-29-generic`, x86-64 |
| Secure Boot | not supported by the system |
| Virtualization | Intel VT-x exposed; system is running on bare metal |
| IPMI | `/dev/ipmi0` exists, but `ipmitool` and `racadm` are absent |

Machine IDs, boot IDs, MAC addresses, disk serials and DIMM serials are
intentionally omitted.

## CPU and NUMA

| Property | Observed value |
|---|---|
| Processors | 2 x Intel Xeon X5650 at 2.67 GHz |
| Topology | 6 cores / 12 threads per socket; 12 cores / 24 threads total |
| NUMA | 2 nodes, one per socket |
| Cache | 24 MiB aggregate L3, 12 MiB per socket |
| SIMD visible in `lscpu` | up to SSE4.2 and AES |
| Missing instruction sets relevant to this project | AVX, F16C and AVX2 |

NUMA node 0 exposes 32,250 MiB and node 1 exposes 30,071 MiB. The remote
kernel reported the expected inter-node distance of 20 versus a local distance
of 10.

The lack of AVX/F16C/AVX2 means the host cannot execute the current exact CPU
attention path or act as a useful drop-in CPU provider for the project's
packed kernels. A different legacy-SSE provider would be new implementation
work and has no demonstrated path to the active throughput targets.

## Memory

| Property | Observed value |
|---|---|
| Installed layout visible through EDAC | 8 x 8 GiB registered DDR3 DIMMs |
| Nominal installed capacity | 64 GiB |
| OS-visible memory | 60.861 GiB |
| Swap | 8 GiB, unused at capture time |
| Memory controllers | three populated channels per socket |
| EDAC counters | zero corrected and zero uncorrected DIMM errors exposed at capture time |

Each socket has two DIMMs on channel 0 and one DIMM on each of channels 1 and
2. DIMM clock and manufacturer data were not available without privileged
`dmidecode`. No RAM bandwidth claim is made because no bandwidth test was
requested or run.

## Accelerators and PCIe

No NVIDIA or AMD compute accelerator is visible in `lspci`. The only display
controller is the embedded Matrox G200eW. `nvidia-smi`, `nvcc` and the NVIDIA
driver/toolkit are absent.

The chipset exposes the following PCIe root-port capabilities through sysfs:

| Root port | Occupancy in the observed tree | Maximum reported link |
|---|---|---:|
| `00:01.0` | dual-port Broadcom NIC | PCIe Gen2 x4 root; endpoints negotiate Gen1 x4 |
| `00:03.0` | PERC 6/i RAID controller | PCIe Gen2 x4 root; endpoint negotiates Gen1 x4 |
| `00:07.0` | no endpoint visible | PCIe Gen2 x8 |
| `00:09.0` | no endpoint visible | PCIe Gen2 x4 |
| `00:0a.0` | no endpoint visible | PCIe Gen2 x4 |

An empty PCIe Gen2 x8 link has an ideal one-direction payload ceiling of about
4 GB/s after 8b/10b encoding, before chipset, NUMA and protocol overhead. This
is a calculation, not a measurement. It is far below the 13.38 GB/s pinned
host-to-device path already measured on the RTX 3090 host.

The presence of an empty root-port link does not prove that a corresponding
physical slot, riser, full-height card clearance or GPU power connector is
available. DMI slot data required privileges and was not accessible.

## Storage

| Property | Observed value |
|---|---|
| Controller | Dell PERC 6/i, reported firmware 1.22 |
| OS-visible disk | `/dev/sda`, 278.9 GiB virtual disk, rotational flag set |
| Root filesystem | ext4 on `/dev/sda2` |
| Root free space | 247.083 GiB |
| EFI partition | 1 GiB FAT32 |
| NVMe | no NVMe controller or namespace visible |
| Optical drive | TSSTcorp DVD+/-RW TS-L633J |

`perccli`, `storcli`, `MegaCli`, `smartctl`, `lsscsi` and `nvme-cli` are not
installed. Consequently, the physical disk count, RAID level, drive health,
cache policy and sustained storage bandwidth remain unknown. The virtual disk
reports itself as rotational, so it must not be documented or treated as
NVMe.

The available capacity could hold some published artifacts, but rotational
RAID plus 1 GbE makes this host unsuitable as a critical remote model store.
The target architecture also explicitly forbids an external expert owner or
outside serving dependency.

## Network

| Interface | Controller | State |
|---|---|---|
| `eno1` | Broadcom NetXtreme II BCM5716 | up, 1,000 Mb/s full duplex, MTU 1500 |
| `eno2` | Broadcom NetXtreme II BCM5716 | down, MTU 1500 |

The active private address is `192.168.100.11/24`, with the default route on
`eno1`. There is no detected 10 GbE adapter. One gigabit Ethernet has a raw
line-rate ceiling of 125 MB/s before protocol overhead.

## Installed software

| Component | Observed state |
|---|---|
| Git | 2.53.0 |
| Python | 3.14.4 |
| GCC / G++ / Clang | absent |
| CMake / Ninja / Make | absent |
| Docker / Podman | absent; Docker service not installed |
| NVIDIA driver / CUDA toolkit | absent |
| SSH service | active; unit reported disabled |

The machine is therefore not currently a build host or a runtime host for this
repository.

## Security observations

The kernel reports IOMMU groups, but passthrough behavior was not tested. CPU
vulnerability status includes:

- MDS vulnerable, with attempted buffer clearing but no matching microcode;
- L1TF mitigations present, while SMT remains vulnerable;
- Spectre v2 mitigations present.

Together with the 2012 BIOS, this prevents treating the machine as an
unqualified multi-tenant production endpoint. No remediation was attempted
during inventory.

## Project suitability

| Proposed role | Verdict | Reason |
|---|---|---|
| Current Qwen/DeepSeek CUDA service | rejected as-is | no CUDA GPU or NVIDIA stack |
| CPU execution tier | rejected for current providers | no AVX, F16C or AVX2; legacy dual-socket DDR3 |
| NVMe expert tier | rejected as-is | no NVMe device; only rotational PERC virtual disk visible |
| Remote expert/model owner | out of architecture | the serving target is self-contained on 3090box |
| General administration/archive node | possible but unqualified | free disk and SSH exist, but RAID health and backup policy are unknown |
| Future GPU retrofit | unqualified | Gen2 x8 is the best empty root link; power, cooling, riser and clearance are unknown |

This classification is based on capacity and capability inspection only. No
throughput result should be attributed to `node10` until the hardware changes
and a goal-specific measurement is explicitly approved.

## Read-only evidence commands

The inventory used ordinary unprivileged commands: `hostnamectl`, `lscpu`,
`free`, `numactl`, EDAC sysfs, PCI sysfs, `lspci`, `lsblk`, `findmnt`, `df`,
`ethtool`, `ip`, `systemctl` and tool-presence checks. Passwordless sudo was
not available, so no privileged hardware inventory was collected.
