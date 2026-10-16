// Word stages on a GPU against the CPU runner, through createProgramRunner and loop-emit.ts's WGSL. Runs under bun with
// Dawn's node bindings ($DAWN_MODULE, the `webgpu` package): Metal on a Mac, Vulkan on Linux.
//   DAWN_MODULE=<webgpu>/index.js bun lab/words/gpu-check.ts
// 1. Random word circuits, one per program: every source a shared word, outputs in a u32 group, 4,096 documents each
//    with its own document index mixed in, compared word for word with the reference evaluator.
// 2. The enumerating loop of tests/words: document d's base-5 digits, at documents past 2^31 (a shared base is added
//    to the store's row), compared with the CPU runner.
import {createDocumentStore, createProgramRunner, requestBatchDevice} from '../../src/json-search/gpu/host.ts'
import {emitLoopWgsl} from '../../src/json-search/loop-emit.ts'
import {type AnyStage, type LoopProgram, createLoopRunner, createWordStage} from '../../src/json-search/loop-program.ts'
import {type Word, WordCircuit} from '../../src/runtime/words.ts'
import {anyWord, digitLoop, randomCircuit, reseed} from '../../tests/words/programs.ts'

const dawn = await import(process.env.DAWN_MODULE ?? 'webgpu') as {create: (flags: string[]) => GPU, globals: Record<string, unknown>}
Object.assign(globalThis, dawn.globals)
Object.defineProperty(globalThis.navigator, 'gpu', {value: dawn.create(process.platform === 'linux' ? ['backend=vulkan'] : []), configurable: true})
const documents = 4096
const {device, adapterInfo} = await requestBatchDevice(documents * 64 * 4) as {device: GPUDevice, adapterInfo: unknown}
console.log(`adapter ${JSON.stringify(adapterInfo)}`)
const store = await createDocumentStore(device, 1, documents)

// 1. Random circuits: sources 0..n-1 are shared words except the last, which is the document index.
reseed(42)
let programs = 0, wordsCompared = 0, mismatches = 0, uncheckedLoads = 0
for (let trial = 0; trial < 60; trial++) {
  const {circuit, words} = randomCircuit(3 + (trial % 6), 10 + 3 * trial)
  const roots = words.slice(-6)
  const last = (circuit.sourceSize - 1) as Word
  const stage = createWordStage({name: 'random', repeat: 'once', circuit, sharedOffset: 0, bindings: new Map([[last, {kind: 'document'}]]),
    outputs: roots.map((wire, column) => ({wire, write: {kind: 'document' as const, group: 0, column}}))})
  const shared = new Float32Array(circuit.sourceSize)
  const sharedWords = new Uint32Array(shared.buffer)
  for (let k = 0; k < circuit.sourceSize - 1; k++) sharedWords[k] = anyWord()
  // Half the programs declare maxima (every shared word small, the document below `documents`), so their loads at a
  // provably small index are written without a bounds check.
  if (trial % 2 === 1) {
    for (let k = 0; k < circuit.sourceSize - 1; k++) { sharedWords[k] = sharedWords[k]! % 8; circuit.assumeAtMost(circuit.source(k), 7) }
    circuit.assumeAtMost(last, documents - 1)
  }
  const loop: LoopProgram<AnyStage> = {positions: 1, buffers: [{name: 'none', cellsPerDocument: 1}], registers: [], fields: [],
    outputGroups: [{name: 'out', kind: 'document', valuesPerDocument: roots.length, fill: 0, type: 'u32'}], stages: [stage], results: [], sharedValues: shared, lengthCell: undefined}
  const module = emitLoopWgsl(loop)
  uncheckedLoads += (module.code.match(/shared_values\[\d+u \+ /g) ?? []).length
  const runner = await createProgramRunner(device, module, shared, store)
  const {outputs} = await runner.evaluate({count: documents, readbacks: [{group: 0, firstDocument: 0, documentCount: documents}]})
  const gpu = new Uint32Array(outputs[0]!.buffer, outputs[0]!.byteOffset, outputs[0]!.length)
  const sources = new Uint32Array(circuit.sourceSize)
  sources.set(sharedWords.subarray(0, circuit.sourceSize - 1))
  for (let d = 0; d < documents; d++) {
    sources[circuit.sourceSize - 1] = d
    const reference = circuit.evaluate(sources)
    roots.forEach((root, column) => { wordsCompared++; if (gpu[d * roots.length + column] !== reference[root]) { if (mismatches++ < 5) console.log(`mismatch: program ${trial}, document ${d}, word ${root} (${JSON.stringify(circuit.operations[root - circuit.sourceSize])}): gpu ${gpu[d * roots.length + column]}, reference ${reference[root]}`) } })
  }
  runner.destroy()
  programs++
}
console.log(`random circuits: ${programs} programs (${uncheckedLoads} loads written without a bounds check), ${wordsCompared.toLocaleString()} words compared, ${mismatches} mismatches`)

// 2. The enumerating loop, its start stage reading a shared base plus the store's row.
const {loop: digits, reference} = digitLoop(14)
const start = new WordCircuit(2)
const based = createWordStage({name: 'start', repeat: 'once', circuit: start, sharedOffset: 7,
  bindings: new Map([[start.source(0), {kind: 'document'}]]), outputs: [{wire: start.add(start.source(0), start.source(1)), write: {kind: 'register', register: 0}}]})
const shared = new Float32Array(9)
shared.set(digits.sharedValues)
const loop: LoopProgram<AnyStage> = {...digits, stages: [based, ...digits.stages.slice(1)], sharedValues: shared}
const runner = await createProgramRunner(device, emitLoopWgsl(loop), shared, store)
const cpu = createLoopRunner(loop, 'generated')
let enumerated = 0, wrong = 0
for (const base of [0, 2 ** 24 - 100, 2 ** 31 + 12345, 0xffffffff - documents + 1]) {
  new Uint32Array(shared.buffer)[8] = base >>> 0
  cpu.setShared(shared)
  const {outputs} = await runner.evaluate({count: documents, shared, readbacks: [{group: 0, firstDocument: 0, documentCount: documents}, {group: 1, firstDocument: 0, documentCount: documents}]})
  const gpuWords = new Uint32Array(outputs[0]!.buffer, outputs[0]!.byteOffset, outputs[0]!.length)
  const out = [new Float32Array(2), new Float32Array(1)], outWords = new Uint32Array(out[0]!.buffer)
  for (let d = 0; d < documents; d++) {
    cpu.run([new Float32Array(1)], d, out, 0, -1)
    const [sum, fours, flag] = reference((base + d) >>> 0)
    enumerated++
    const ok = gpuWords[2 * d] === sum && gpuWords[2 * d + 1] === fours && outputs[1]![d] === flag && outWords[0] === sum && outWords[1] === fours && out[1]![0] === flag
    if (!ok && wrong++ < 5) console.log(`enumeration mismatch at ${base + d}: gpu ${gpuWords[2 * d]} ${gpuWords[2 * d + 1]} ${outputs[1]![d]}, cpu ${outWords[0]} ${outWords[1]} ${out[1]![0]}, reference ${sum} ${fours} ${flag}`)
  }
}
runner.destroy()
console.log(`enumeration: ${enumerated.toLocaleString()} documents at four bases up to 2^32 - 1, ${wrong} mismatches (GPU, generated JS and the reference agree)`)
process.exit(mismatches + wrong === 0 ? 0 : 1)
