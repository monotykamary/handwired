// A program written as loops over one document's positions, the form of docs/patterns.md's "A loop over items that mixes
// per-item work with values carried to the next item": the work of one position is ordinary builder code, compiled once
// as its own small program, and it runs once per position. JS for what a looped stage does:
//
//   for (let position = 0; position < positions; position++) {
//     const values = step(codePoints[position], carried)   // the compiled step, e.g. 37 operations
//     carried = values.next                                // carried to the next position
//     perPosition[position] = values.kept                  // kept for a later loop, or for the page
//   }
//
// What a step reads and writes besides its own shared cells, e.g. a pattern's tables:
//   item       the document's cell at the loop's position, e.g. the code point
//   cell       one cell of the document at a fixed column, e.g. the text's length
//   register   one number carried from a position to the next, with an initial value, e.g. a state machine's state
//   field      one number per position, written by one loop and read by a later one, e.g. the reverse pass's bits that
//              the forward pass reads
//   outputs    a document's results after the last stage, and per position values for the documents a page shows
//
// The same LoopProgram runs three ways, and the tests compare all three with plain JavaScript references:
// createLoopRunner on the CPU executor, createLoopRunner on generated JavaScript, and loop-emit.ts's WGSL, where each
// stage's statements are written once inside a `for` loop.
//
// Word stages (runtime/words.ts): a stage may be a WordCircuit instead of a Circuit, whose values are exact u32s. It
// reads the same bindings, plus two only words can hold exactly: the document's index (`document`, e.g. which of 2^32
// candidates this invocation enumerates) and the loop's position (`position`). Its unbound source cells are shared
// words, whose bits sit in the shared values (written through a Uint32Array view: copies of a Float32Array keep the
// bits). Registers and output groups say whether they hold f32 or u32; where a value crosses between kinds it is
// converted, f32(word) or u32(value) for a value that is a whole number from 0.
import {type Circuit, type Wire, invariant} from '../runtime/numeric-ui-core.ts'
import {type Word, WordCircuit} from '../runtime/words.ts'
import {type TensorProgram, positionsOfWires} from './emit.ts'
import {compileChunks} from './generated-js.ts'

// Where the value of one source cell of a stage's circuit comes from. A source cell without a binding is a shared cell.
export type Binding =
  | {kind: 'item', buffer: number}
  | {kind: 'cell', buffer: number, column: number}
  | {kind: 'register', register: number}
  | {kind: 'field', field: number, component: number}
// A word stage reads those too, and two values only a word holds exactly: the document's index (params.first + the
// invocation's number on the GPU) and the loop's position.
export type WordBinding = Binding | {kind: 'document'} | {kind: 'position'}

// Where one output of a stage goes.
type Write =
  | {kind: 'register', register: number}
  | {kind: 'field', field: number, component: number}
  // One value per position in an output group of kind 'position'.
  | {kind: 'position', group: number}
  // One value per document in an output group of kind 'document'. Only a stage that runs once writes these.
  | {kind: 'document', group: number, column: number}

export type Stage = {
  name: string
  // 'once' runs the program one time. 'forward' runs it for positions 0, 1, 2, and so on, and 'backward' from the last
  // position down to 0.
  repeat: 'once' | 'forward' | 'backward'
  circuit: Circuit
  // Compiled for exactly the wires of `outputs`.
  program: TensorProgram
  // Where this stage's source table starts in the loop program's shared values.
  sharedOffset: number
  bindings: ReadonlyMap<Wire, Binding>
  outputs: readonly {wire: Wire, write: Write}[]
  kind?: 'float'
}

// A stage of u32 words. Same shape, without a compiled program: the WordCircuit's statements are the program.
export type WordStage = {
  kind: 'words'
  name: string
  repeat: 'once' | 'forward' | 'backward'
  circuit: WordCircuit
  sharedOffset: number
  bindings: ReadonlyMap<Word, WordBinding>
  // `stop`: a looped word stage ends its loop after a position where this word is nonzero (its other writes of that
  // position happen first), e.g. when an enumeration has run out of candidates. Positions after it get nothing, as
  // positions past a text's end; a 'position' output group's later positions keep the group's fill only when the
  // loop program has a lengthCell, so a stage that stops writes no 'position' outputs.
  outputs: readonly {wire: Word, write: Write | {kind: 'stop'}}[]
}
export type AnyStage = Stage | WordStage
export const isWordStage = (stage: AnyStage): stage is WordStage => stage.kind === 'words'
export type ValueType = 'f32' | 'u32'

// S: the kinds of stage it holds; LoopProgram<AnyStage> may hold word stages, and LoopProgram holds FP32 stages only.
export type LoopProgram<S extends AnyStage = Stage> = {
  // How many positions the looped stages run, e.g. 512 code points.
  positions: number
  // The buffers of per-document cells that the stages read, e.g. buffer 0 with 513 cells per document.
  buffers: readonly {name: string, cellsPerDocument: number}[]
  // `type` 'u32' holds a word; `initial` is then a u32.
  registers: readonly {name: string, initial: number, type?: ValueType}[]
  // `bits` says every value of the field is 0 or 1, so the WGSL keeps a position's values in one 32-bit word.
  fields: readonly {name: string, width: number, bits: boolean}[]
  // A 'document' group holds valuesPerDocument values for every document. A 'position' group holds one value per
  // position, for the documents of a window that the caller picks, e.g. the ones on screen. `fill` is what a position
  // past the text's end gets when the loops stop there.
  // `type` 'u32': the group's buffer holds the bits of u32s, read through a Uint32Array view.
  outputGroups: readonly {name: string, kind: 'document' | 'position', valuesPerDocument: number, fill: number, type?: ValueType}[]
  stages: readonly S[]
  // After the last stage, these registers are written as a document's outputs.
  results: readonly {group: number, column: number, register: number}[]
  // Every stage's source table, one after the other, with the values a new evaluation starts from.
  sharedValues: Float32Array
  // The cell that holds the text's length in code points. With it, the looped stages run only the positions before the
  // text's end, which is only right for steps that leave every register as it is at a position past the end.
  lengthCell: {buffer: number, column: number} | undefined
}

// Compiles one stage. `outputs` may name a wire more than once, e.g. a bit that is both kept and carried.
export function createStage(description: Omit<Stage, 'program'>): Stage {
  const program = description.circuit.compileTensorProgram([...new Set(description.outputs.map(output => output.wire))])
  for (const wire of description.bindings.keys()) invariant(wire < program.sourceSize, 'a binding names a source cell')
  for (const {wire, write} of description.outputs) {
    invariant(wire >= program.sourceSize, `${description.name}: an output is a computed value, not a source cell`)
    invariant(write.kind !== 'document' || description.repeat === 'once', 'only a stage that runs once writes a document\'s outputs')
    invariant(write.kind !== 'position' || description.repeat !== 'once', 'only a looped stage writes per position')
  }
  for (const binding of description.bindings.values()) {
    invariant(description.repeat !== 'once' || (binding.kind !== 'item' && binding.kind !== 'field'), 'a stage that runs once reads no item and no field')
  }
  return {...description, program}
}

// Compiles nothing: checks a word stage's description.
export function createWordStage(description: Omit<WordStage, 'kind'>): WordStage {
  const {circuit} = description
  for (const word of description.bindings.keys()) invariant(word < circuit.sourceSize, 'a binding names a source cell')
  const stops = description.outputs.filter(output => output.write.kind === 'stop').length
  invariant(stops <= 1 && (stops === 0 || description.repeat !== 'once'), 'a looped word stage has at most one stop')
  invariant(stops === 0 || description.outputs.every(output => output.write.kind !== 'position'), 'a stage that stops writes no position outputs')
  for (const {wire, write} of description.outputs) {
    invariant(wire < circuit.size, `${description.name}: an output is a word of the circuit`)
    invariant(write.kind !== 'document' || description.repeat === 'once', 'only a stage that runs once writes a document\'s outputs')
    invariant(write.kind !== 'position' || description.repeat !== 'once', 'only a looped stage writes per position')
  }
  for (const binding of description.bindings.values()) {
    invariant(description.repeat !== 'once' || (binding.kind !== 'item' && binding.kind !== 'field' && binding.kind !== 'position'), 'a stage that runs once reads no item, field or position')
  }
  // A loaded table is read by index from the shared words, so its cells are shared, not bound.
  const tables = new Set<string>()
  for (const operation of circuit.operations) {
    if (operation.op !== 'load' || tables.has(`${operation.table}:${operation.length}`)) continue
    tables.add(`${operation.table}:${operation.length}`)
    for (const word of description.bindings.keys()) invariant(!(word >= operation.table && word < operation.table + operation.length), `${description.name}: a loaded table's cells are shared words, and cell ${word} is bound`)
  }
  return {kind: 'words', ...description}
}

// The u32 a value of the other kind becomes: the whole part of a number from 0, as WGSL's u32(f32) converts it.
export const toWord = (value: number) => (value >= 0 ? Math.min(Math.trunc(value), 0xffffffff) >>> 0 : 0)

export function describeLoop(loop: LoopProgram<AnyStage>) {
  return loop.stages.map(stage => isWordStage(stage) ? ({
    name: stage.name, repeat: stage.repeat, operations: stage.circuit.operations.length, layers: 0, indexReads: 0,
  }) : ({
    name: stage.name, repeat: stage.repeat,
    operations: stage.program.valueCount - stage.program.sourceSize,
    layers: stage.program.layers.length,
    indexReads: stage.program.layers.reduce((total, layer) => total + layer.indexReads.length, 0),
  }))
}

// How many operations one document takes when every looped stage runs every position.
/** @public */
export function operationsPerDocument(loop: LoopProgram<AnyStage>) {
  return describeLoop(loop).reduce((total, stage) => total + stage.operations * (stage.repeat === 'once' ? 1 : loop.positions), 0)
}

export type LoopRunner = ReturnType<typeof createLoopRunner>

// Runs a loop program on the CPU, one document per call. With 'executor', every stage runs on the CPU executor, the
// repo's definition of a program's values. With 'generated', a stage's operations are JavaScript statements, see
// generated-js.ts, which gives the same values several times faster. A word stage runs its WordCircuit's reference
// evaluator under 'executor' and its compiled JavaScript under 'generated'.
export function createLoopRunner(loop: LoopProgram<AnyStage>, backend: 'executor' | 'generated') {
  const {positions} = loop
  // Registers of both kinds share one buffer: an f32 register through `registers`, a u32 one through `registerWords`.
  const registerBuffer = new ArrayBuffer(loop.registers.length * 4)
  const registers = new Float32Array(registerBuffer)
  const registerWords = new Uint32Array(registerBuffer)
  const wordRegister = loop.registers.map(register => register.type === 'u32')
  const wordGroup = loop.outputGroups.map(group => group.type === 'u32')
  const fields = loop.fields.map(field => new Float32Array(positions * field.width))
  const sharedWords = new Uint32Array(loop.sharedValues.buffer, loop.sharedValues.byteOffset, loop.sharedValues.length)
  // A register's value as the reading stage's kind sees it.
  const readFloat = (register: number) => (wordRegister[register] ? Math.fround(registerWords[register]!) : registers[register]!)
  const readWord = (register: number) => (wordRegister[register] ? registerWords[register]! : toWord(registers[register]!))
  const writeRegister = (register: number, value: number, fromWord: boolean) => {
    if (wordRegister[register]) registerWords[register] = fromWord ? value : toWord(value)
    else registers[register] = fromWord ? Math.fround(value) : value
  }
  const writeOutput = (outputs: readonly Float32Array[], group: number, at: number, value: number, fromWord: boolean) => {
    if (wordGroup[group]) new Uint32Array(outputs[group]!.buffer, outputs[group]!.byteOffset, outputs[group]!.length)[at] = fromWord ? value : toWord(value)
    else outputs[group]![at] = fromWord ? Math.fround(value) : value
  }

  type Bound = {position: number, binding: WordBinding}
  const stages = loop.stages.map(stage => {
    const bound: Bound[] = [...(stage.bindings as ReadonlyMap<number, WordBinding>)].map(([wire, binding]) => ({position: wire, binding}))
    if (isWordStage(stage)) {
      const {circuit} = stage
      const values = new Uint32Array(circuit.size)
      const roots = stage.outputs.map(output => output.wire)
      // Unbound source cells are shared words.
      const boundSet = new Set(bound.map(entry => entry.position))
      const shared = Array.from({length: circuit.sourceSize}, (_, cell) => cell).filter(cell => !boundSet.has(cell))
      const run = backend === 'generated' ? circuit.compileJs(roots) : undefined
      const compute = () => {
        if (run) run(values)
        else values.set(circuit.evaluate(values.subarray(0, circuit.sourceSize)))
        return values
      }
      return {kind: 'words' as const, stage, values, compute, bound, shared, writes: stage.outputs.map(output => ({at: output.wire as number, write: output.write}))}
    }
    const {program} = stage
    const outputPositions = positionsOfWires(program, stage.outputs.map(output => output.wire))
    // `values` starts with the stage's source table. `compute` returns the array that holds the outputs, at `outputAt`.
    let values: Float32Array
    let compute: () => Float32Array
    let outputAt: number[]
    if (backend === 'executor') {
      const evaluate = stage.circuit.createTensorProgramEvaluator(program)
      values = new Float32Array(program.sourceSize)
      compute = () => evaluate(values)
      outputAt = stage.outputs.map(output => program.roots.indexOf(output.wire))
    } else {
      const {chunks} = compileChunks(program)
      const all = new Float32Array(program.valueCount)
      values = all
      compute = () => {
        for (let index = 0; index < chunks.length; index++) chunks[index]!(all)
        return all
      }
      outputAt = outputPositions
    }
    return {kind: 'float' as const, stage, values, compute, bound, shared: [] as number[], writes: stage.outputs.map((output, index) => ({at: outputAt[index]!, write: output.write}))}
  })

  // The shared values, e.g. after writing a new pattern's cells into them.
  function setShared(shared: Float32Array) {
    invariant(shared.length === loop.sharedValues.length, 'the shared values have the loop program\'s length')
    const words = new Uint32Array(shared.buffer, shared.byteOffset, shared.length)
    for (const entry of stages) {
      if (entry.kind === 'float') entry.values.set(shared.subarray(entry.stage.sharedOffset, entry.stage.sharedOffset + entry.stage.program.sourceSize))
      else for (const cell of entry.shared) entry.values[cell] = words[entry.stage.sharedOffset + cell]!
    }
  }
  setShared(loop.sharedValues)
  void sharedWords

  // Runs document `row` of the buffers. The document's outputs go to row `documentRow` of each 'document' group, and
  // with windowRow >= 0, the per position values go to row `windowRow` of each 'position' group.
  function run(buffers: readonly Float32Array[], row: number, outputs: readonly Float32Array[], documentRow: number, windowRow: number) {
    const bases = loop.buffers.map(buffer => row * buffer.cellsPerDocument)
    loop.registers.forEach((register, index) => {
      if (wordRegister[index]) registerWords[index] = register.initial
      else registers[index] = register.initial
    })
    const limit = loop.lengthCell === undefined ? positions : Math.min(positions, buffers[loop.lengthCell.buffer]![bases[loop.lengthCell.buffer]! + loop.lengthCell.column]!)
    for (const entry of stages) {
      const {stage, values, compute, bound, writes} = entry
      const words = entry.kind === 'words'
      const count = stage.repeat === 'once' ? 1 : limit
      for (let step = 0; step < count; step++) {
        const position = stage.repeat === 'backward' ? limit - 1 - step : step
        for (let index = 0; index < bound.length; index++) {
          const {position: at, binding} = bound[index]!
          let value: number
          switch (binding.kind) {
            case 'item': value = buffers[binding.buffer]![bases[binding.buffer]! + position]!; break
            case 'cell': value = buffers[binding.buffer]![bases[binding.buffer]! + binding.column]!; break
            case 'field': value = fields[binding.field]![position * loop.fields[binding.field]!.width + binding.component]!; break
            case 'register': value = words ? readWord(binding.register) : readFloat(binding.register); values[at] = value; continue
            case 'document': value = row; break
            case 'position': value = position; break
          }
          values[at] = words ? toWord(value) : value
        }
        const computed = compute()
        let stop = false
        for (let index = 0; index < writes.length; index++) {
          const {at, write} = writes[index]!
          const value = computed[at]!
          switch (write.kind) {
            case 'stop':
              stop = value !== 0
              break
            case 'register':
              writeRegister(write.register, value, words)
              break
            case 'field':
              fields[write.field]![position * loop.fields[write.field]!.width + write.component] = words ? Math.fround(value) : value
              break
            case 'position':
              if (windowRow >= 0) writeOutput(outputs, write.group, windowRow * positions + position, value, words)
              break
            case 'document':
              writeOutput(outputs, write.group, documentRow * loop.outputGroups[write.group]!.valuesPerDocument + write.column, value, words)
              break
          }
        }
        if (stop) break
      }
    }
    if (windowRow >= 0) {
      loop.outputGroups.forEach((group, index) => {
        if (group.kind === 'position') outputs[index]!.fill(group.fill, windowRow * positions + limit, (windowRow + 1) * positions)
      })
    }
    for (const result of loop.results) {
      const at = documentRow * loop.outputGroups[result.group]!.valuesPerDocument + result.column
      if (wordRegister[result.register]) writeOutput(outputs, result.group, at, registerWords[result.register]!, true)
      else writeOutput(outputs, result.group, at, registers[result.register]!, false)
    }
  }

  return {setShared, run}
}
