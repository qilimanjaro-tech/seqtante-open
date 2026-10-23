---
status: proposed
---

# The redesign lands on one branch, in small PRs, behind a parser-first test net

ADRs 0001 and 0003 to 0006 together change the calibration file, the tree format, the parser, the graph and every node. Tests today cover the fits, through the fitting harness, and three of the four nodes; nothing covers `CalibrationNode`, `CalibrationParser` or `CalibrationGraph`. The failure this redesign risks is a silent parameter-mapping error that shows up as wrong physics hours into a fridge run, and reading a large diff carefully does not catch it. We decided to land the redesign on one long-lived branch off `main`, through small PRs into that branch, building the whole stack for one step first.

- First vertical slice: T1 at the sweet spot, written directly against the new base. QHC-1596 restarts from this design, reusing parts of its current branch. The slice exercises operating-point application, settings resolution, the three phases and a benchmark verdict, without porting an existing node first.
- Second slice: `two_tone_vs_flux` as a `FluxSweepNode`, since `sweet_spot_flux` is the root of every branch of the tree.
- During the port, `NODE_REGISTRY` accepts both legacy functions and node classes, and the graph branches once on which it got.
- Two kinds of test: exhaustive coverage of the parser and settings resolution, and one test per node that mocks the hardware loop and asserts the arguments it receives.
- A fix that changes measured values lands as its own PR ahead of the redesign, not inside it, so it gets hardware sign-off instead of being reviewed as part of a refactor.

## Considered options

- **One big PR**: no dual registry and one review, but a change across every node is unreviewable in practice, and `git bisect` stops helping once something goes wrong on hardware.
- **Strangler on `main`, both APIs supported long term**: `main` stays shippable, but the dual registry becomes semi-permanent, and the tree-format change (ADR-0003) cannot land in pieces on `main` without breaking operators mid-migration.

## Consequences

- The parser is pure (tree in, `dict[Target, Settings]` out, no platform, hardware or database), so the riskiest layer is also the cheapest to test.
- Per-node call-assertion tests catch the rename and mapping bug class, including a misspelt key falling back to a default.
- Node-by-node PRs keep `git bisect` meaningful.
- The branch needs rebasing against `main`, and the registry holds two shapes for a while.

## Open questions

- **Scope for the December delivery.** Which of ADRs 0004 to 0006 land before it and which follow; effort per proposal is to be estimated first.
