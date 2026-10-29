# Contributing to VDWT

Thanks for helping. Bug reports, results on your own footage, benchmarks on new hardware and pull
requests are all welcome.

## Reporting a problem

Open an issue with:

- the exact command line,
- the output of `vdwt info` (and `vdwt selftest` if a kernel might be involved),
- if possible, a short clip that shows the problem (a few frames are enough).

## Pull requests

- Run `cargo test --release` and `cargo fmt` before sending.
- The bitstream must stay identical for every thread count and on every platform; `vdwt selftest`
  and the test suite check this.
- For compression changes, include a before/after BD-rate (YUV-PSNR 6:1:1) on a few clips.

## Contributor License Agreement

VDWT is offered under the AGPL-3.0, the PolyForm Small Business License and a commercial license.
To keep all three possible, every contribution needs the contributor's agreement to the
[Contributor License Agreement](CLA.md). Add this line to your pull request description:

> I have read the CLA in CLA.md and I agree to it for this contribution.

Pull requests without it cannot be merged.
