// Words: the reference evaluator against the compiled JavaScript on random programs, and word stages in loop programs
// against plain JavaScript, on both CPU backends. lab/words/gpu-check.ts runs the same loop programs' WGSL on a GPU.
import {expect, test} from 'bun:test'
import {applyWord} from '../../src/runtime/words.ts'
import {createLoopRunner} from '../../src/json-search/loop-program.ts'
import {emitLoopWgsl} from '../../src/json-search/loop-emit.ts'
import {anyWord, digitLoop, rand, randomCircuit} from './programs.ts'

test('applyWord is u32 arithmetic with the stated rules at zero and for shifts', () => {
  expect(applyWord('sub', 0, 1)).toBe(0xffffffff)
  expect(applyWord('add', 0xffffffff, 2)).toBe(1)
  expect(applyWord('mul', 0x10000, 0x10000)).toBe(0)
  expect(applyWord('monus', 3, 5)).toBe(0)
  expect(applyWord('div', 7, 0)).toBe(0)
  expect(applyWord('mod', 7, 0)).toBe(7)
  expect(applyWord('shl', 1, 33)).toBe(2)
  expect(applyWord('shr', 0x80000000, 31)).toBe(1)
  expect(applyWord('not', 0)).toBe(0xffffffff)
  expect(applyWord('select', 0, 5, 9)).toBe(9)
  for (let k = 0; k < 2000; k++) { const a = anyWord(), b = anyWord(); expect(applyWord('add', applyWord('mul', b, applyWord('div', a, b)), applyWord('mod', a, b))).toBe(a) }
})

test('compiled JavaScript gives the reference evaluator\'s words on random programs', () => {
  for (let trial = 0; trial < 300; trial++) {
    const {circuit, words} = randomCircuit(2 + rand(6), 5 + rand(60))
    const roots = words.slice(-1 - rand(5))
    const run = circuit.compileJs(roots)
    for (let input = 0; input < 20; input++) {
      const sources = Uint32Array.from({length: circuit.sourceSize}, anyWord)
      const reference = circuit.evaluate(sources)
      const values = new Uint32Array(circuit.size); values.set(sources)
      run(values)
      for (const root of roots) expect(values[root]).toBe(reference[root])
    }
  }
})

test('word stages run in loop programs on both CPU backends, with an FP32 stage reading a u32 register', () => {
  const {loop, reference} = digitLoop(14)
  for (const backend of ['executor', 'generated'] as const) {
    const runner = createLoopRunner(loop, backend)
    const outputs = [new Float32Array(2), new Float32Array(1)]
    const words = new Uint32Array(outputs[0]!.buffer)
    for (const d of [0, 1, 4, 24, 5 ** 13 * 3 + 4, 2 ** 31 + 7, 0xffffffff, ...Array.from({length: 200}, () => anyWord())]) {
      runner.run([new Float32Array(1)], d, outputs, 0, -1)
      const [sum, fours, flag] = reference(d)
      expect([words[0], words[1], outputs[1]![0]]).toEqual([sum, fours, flag])
    }
  }
})

test('the WGSL declares words where words are held', () => {
  const {code} = emitLoopWgsl(digitLoop(14).loop)
  expect(code).toContain('var<storage, read> shared_values: array<u32>')
  expect(code).toContain('var r0: u32 = 0u;')
  expect(code).toContain('instance_outputs_0: array<u32>')
  expect(code).toContain('instance_outputs_1: array<f32>')
  expect(code).toMatch(/let row: u32 = params.first \+ i;/)
  expect(code).toContain('for (var j: u32 = 0u; j < 14u; j++) {') // a fixed count is written in the loop's header
})

test('a looped word stage stops after the position where its stop word is nonzero', async () => {
  const {WordCircuit} = await import('../../src/runtime/words.ts')
  const {createWordStage} = await import('../../src/json-search/loop-program.ts')
  // Count positions until the count reaches the document's index (at most 40 positions).
  const w = new WordCircuit(2)
  const count = w.source(0), doc = w.source(1)
  const next = w.add(count, w.constant(1))
  const stage = createWordStage({name: 'count', repeat: 'forward', circuit: w, sharedOffset: 0,
    bindings: new Map([[count, {kind: 'register', register: 0}], [doc, {kind: 'document'}]]),
    outputs: [{wire: next, write: {kind: 'register', register: 0}}, {wire: w.le(doc, next), write: {kind: 'stop'}}]})
  const loop = {positions: 40, buffers: [{name: 'none', cellsPerDocument: 1}], registers: [{name: 'count', initial: 0, type: 'u32' as const}], fields: [],
    outputGroups: [{name: 'count', kind: 'document' as const, valuesPerDocument: 1, fill: 0, type: 'u32' as const}], stages: [stage], results: [{group: 0, column: 0, register: 0}],
    sharedValues: new Float32Array(2), lengthCell: undefined}
  for (const backend of ['executor', 'generated'] as const) {
    const runner = createLoopRunner(loop, backend)
    const out = [new Float32Array(1)], words = new Uint32Array(out[0]!.buffer)
    for (const d of [0, 1, 5, 39, 40, 41, 1000]) { runner.run([new Float32Array(1)], d, out, 0, -1); expect(words[0]).toBe(Math.min(Math.max(d, 1), 40)) }
  }
  expect(emitLoopWgsl(loop).code).toContain('if (a_stop) { break; }')
})

test('inferred maxima bound every value, and in-range loads drop their bounds check', async () => {
  const {WordCircuit} = await import('../../src/runtime/words.ts')
  for (let trial = 0; trial < 300; trial++) {
    const {circuit, words} = randomCircuit(4 + rand(4), 10 + rand(50))
    const declared = Array.from({length: circuit.sourceSize}, () => (rand(2) ? rand(64) : anyWord()))
    declared.forEach((maximum, cell) => circuit.assumeAtMost(circuit.source(cell), maximum))
    const run = circuit.compileJs(words)
    for (let input = 0; input < 30; input++) {
      const sources = Uint32Array.from(declared, maximum => (rand(4) === 0 ? maximum : Math.floor((rand(2 ** 30) / 2 ** 30) * (maximum + 1)) >>> 0))
      const reference = circuit.evaluate(sources)
      const values = new Uint32Array(circuit.size); values.set(sources)
      run(values)
      for (const word of words) { expect(reference[word]! <= circuit.maximum(word)).toBe(true); expect(values[word]).toBe(reference[word]) }
    }
  }
  const w = new WordCircuit(9)
  const position = w.source(8)
  w.assumeAtMost(position, 7)
  const read = w.load(0, 8, position), unproven = w.load(0, 8, w.add(position, w.constant(1)))
  const {body} = w.writeWgsl([read, unproven], {prefix: 'a_', source: cell => `shared_values[${cell}u]`})
  expect(body.join('\n')).toContain('= shared_values[0u + shared_values[8u]];')
  expect(body.filter(line => line.includes('select(0u')).length).toBe(1)
  expect(() => w.evaluate(Uint32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 8]))).toThrow()
})
