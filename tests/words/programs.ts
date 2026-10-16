// Programs the words tests and lab/words/gpu-check.ts share: random word circuits, and a loop program that enumerates.
import {Circuit, emitCompare} from '../../src/runtime/numeric-ui-core.ts'
import {type Word, type WordOp, WordCircuit} from '../../src/runtime/words.ts'
import {type AnyStage, type LoopProgram, createStage, createWordStage} from '../../src/json-search/loop-program.ts'

let seed = 1
export const reseed = (value: number) => { seed = value }
export const rand = (n: number) => { seed = (seed * 1103515245 + 12345) % 2 ** 31; return Math.floor((seed / 2 ** 31) * n) }
export const edge = [0, 1, 2, 3, 31, 32, 33, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff, 2 ** 24, 2 ** 24 + 1]
export const anyWord = () => (rand(3) === 0 ? edge[rand(edge.length)]! : rand(3) === 0 ? rand(64) : Math.floor((rand(2 ** 30) * 4 + rand(4))) >>> 0)
const ops: Exclude<WordOp, 'load'>[] = ['add', 'sub', 'mul', 'monus', 'div', 'mod', 'min', 'max', 'and', 'or', 'xor', 'shl', 'shr', 'eq', 'ne', 'lt', 'le', 'not', 'select']

export function randomCircuit(sources: number, operations: number) {
  const circuit = new WordCircuit(sources)
  const words: Word[] = Array.from({length: sources}, (_, index) => circuit.source(index))
  for (let k = 0; k < operations; k++) {
    const pick = () => words[rand(words.length)]!
    const r = rand(20)
    if (r === 0) words.push(circuit.constant(anyWord()))
    else if (r === 1 && sources >= 4) words.push(circuit.load(0, sources - 1, pick())) // cells 0..n-2: the last may be bound
    else {
      const op = ops[rand(ops.length)]!
      words.push(op === 'not' ? circuit.op(op, pick()) : op === 'select' ? circuit.op(op, pick(), pick(), pick()) : circuit.op(op, pick(), pick()))
    }
  }
  return {circuit, words}
}

// Enumeration: document d is a sequence of `positions` digits in base 5 (d's digits, lowest first); the stage adds each
// digit times its position into a u32 register and counts the digits equal to 4. Only words hold d exactly past 2^24.
export function digitLoop(positions: number) {
  const words = new WordCircuit(5)
  const [rest, sum, fours, position, base] = [0, 1, 2, 3, 4].map(index => words.source(index)) as [Word, Word, Word, Word, Word]
  const digit = words.mod(rest, base)
  const stage = createWordStage({
    name: 'digits', repeat: 'forward', circuit: words, sharedOffset: 0,
    bindings: new Map([[rest, {kind: 'register', register: 0}], [sum, {kind: 'register', register: 1}], [fours, {kind: 'register', register: 2}], [position, {kind: 'position'}]]),
    outputs: [
      {wire: words.div(rest, base), write: {kind: 'register', register: 0}},
      {wire: words.add(sum, words.mul(digit, position)), write: {kind: 'register', register: 1}},
      {wire: words.add(fours, words.eq(digit, words.constant(4))), write: {kind: 'register', register: 2}},
    ],
  })
  const start = new WordCircuit(1)
  const first = createWordStage({name: 'start', repeat: 'once', circuit: start, sharedOffset: 5, bindings: new Map([[start.source(0), {kind: 'document'}]]),
    outputs: [{wire: start.source(0), write: {kind: 'register', register: 0}}]})
  // An FP32 stage after the words: is the count of fours at least 2? It reads the u32 register as f32.
  const floats = new Circuit(1)
  floats.assumeRange(floats.source(0), 0, positions)
  floats.assumeGrid(floats.source(0), 1)
  const atLeastTwo = emitCompare(floats, floats.source(0), '>=', floats.constant(2))
  const last = createStage({name: 'compare', repeat: 'once', circuit: floats, sharedOffset: 6, bindings: new Map([[floats.source(0), {kind: 'register', register: 2}]]),
    outputs: [{wire: atLeastTwo, write: {kind: 'register', register: 3}}]})
  // Each stage's source table, one after the other: the digits stage's 5 cells (the base, a shared word, is cell 4),
  // then the start's 1 and the comparison's 1, all bound.
  const shared = new Float32Array(7)
  new Uint32Array(shared.buffer)[4] = 5
  const loop: LoopProgram<AnyStage> = {
    positions, buffers: [{name: 'none', cellsPerDocument: 1}],
    registers: [{name: 'rest', initial: 0, type: 'u32'}, {name: 'sum', initial: 0, type: 'u32'}, {name: 'fours', initial: 0, type: 'u32'}, {name: 'two fours', initial: 0}],
    fields: [], outputGroups: [{name: 'out', kind: 'document', valuesPerDocument: 2, fill: 0, type: 'u32'}, {name: 'flag', kind: 'document', valuesPerDocument: 1, fill: 0}],
    stages: [first, stage, last], results: [{group: 0, column: 0, register: 1}, {group: 0, column: 1, register: 2}, {group: 1, column: 0, register: 3}],
    sharedValues: shared, lengthCell: undefined,
  }
  const reference = (d: number) => { let rest = d, sum = 0, fours = 0; for (let p = 0; p < positions; p++) { const digit = rest % 5; rest = Math.floor(rest / 5); sum = (sum + digit * p) >>> 0; fours += digit === 4 ? 1 : 0 } return [sum, fours, fours >= 2 ? 1 : 0] }
  return {loop, reference}
}

