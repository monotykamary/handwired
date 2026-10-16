# Words (the u32-words fork)

A circuit computes over FP32: whole numbers are exact up to 2^24, every operation is affine + ReGLU, and that's what makes
a program differentiable. Some programs are discrete all the way down and need more than 2^24: a counter, an id, a
balance in cents, or the index of one candidate among 2^32 that an enumeration visits. Words are their half: a
`WordCircuit` (src/runtime/words.ts) computes u32s, exactly, and nothing in it is differentiated.

## Operations

`add`, `sub`, `mul` wrap as u32 arithmetic does; `monus` is a ∸ b (a − b when b ≤ a, else 0); `div` and `mod` (by zero:
0 and the dividend, so a = b × (a / b) + a % b always holds); `min`, `max`; `and`, `or`, `xor`, `not`, `shl`, `shr` (by
the low 5 bits of the amount); `eq`, `ne`, `lt`, `le` (1 or 0); `select(c, a, b)` (c ≠ 0 ? a : b); `constant`; and
`load(table, length, index)`, the source cell at table + index, or 0 past the table. Identical operations share one
word. `applyWord` and `evaluate` are the definition; `compileJs` and `writeWgsl` must give the same u32 for every input,
and write the rules WGSL leaves to the platform (division by zero, shift amounts) out explicitly.

## In loop programs

A stage of a loop program may be a word stage (`createWordStage`): the same bindings as an FP32 stage, plus two only a
word holds exactly, `{kind: 'document'}` (the document's index, `params.first` + the invocation's number on the GPU) and
`{kind: 'position'}`. Its unbound source cells are shared words, whose bits sit in the shared values (written through a
Uint32Array view; copying a Float32Array keeps the bits, and the GPU runner uploads bytes). A looped word stage may
write `{kind: 'stop'}`: the loop ends after a position where that word is nonzero, which is how an enumeration that
visits several candidates per document moves on as soon as one is done.

Registers and output groups say `type: 'u32'` to hold words. Where a value crosses kinds it is converted: f32(word),
exact below 2^24, or u32(value), the whole part of a value from 0. `LoopProgram<AnyStage>` may hold word stages;
`LoopProgram` alone still means FP32 stages only, so existing programs and their writers are unchanged.

| Backend | Word stages |
| --- | --- |
| `createLoopRunner(loop, 'executor')` | the reference evaluator |
| `createLoopRunner(loop, 'generated')` | `compileJs` |
| `emitLoopWgsl` (loop-emit.ts) | `writeWgsl`: `let a_w12: u32 = ...`; `shared_values` becomes `array<u32>` and FP32 stages read it with `bitcast<f32>`. A loop over a fixed count (no lengthCell) has the count in its header, for every program: Metal's compiler treats a `let` bound as unknown, a third of the time of a 2^31-case enumeration |
| `emitBitLoopWgsl` (bit-emit.ts) | FP32 programs only: a word stage is integer code already |
| `createProgramRunner` | unchanged: it binds buffers and copies bytes |

## Checks

- `tests/words/words.test.ts`: the reference against compiled JavaScript on 300 random programs (edge values included),
  and a loop program that enumerates (a document's base-5 digits past 2^31, with an FP32 stage reading a u32 register)
  on both CPU backends.
- `lab/words/gpu-check.ts`: the same programs' WGSL through `createProgramRunner` on a GPU (Dawn's node bindings:
  Metal on a Mac, Vulkan on Linux), every output word against the reference: 1,474,560 words over 60 random programs,
  and the enumeration at four bases up to 2^32 − 1, with no mismatch on an Apple M2.
- The json-search suite (184 tests) passes unchanged.

```sh
bun test tests/words
DAWN_MODULE=<webgpu package>/index.js bun lab/words/gpu-check.ts
```
