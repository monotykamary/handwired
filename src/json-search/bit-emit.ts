// A second WGSL writer for loop programs. It takes the same LoopProgram as loop-emit.ts and writes the same shader
// around it: one invocation per document, a `for` loop per looped stage, the same buffers and parameters. What differs
// is how one operation is written. loop-emit.ts writes every operation as f32 arithmetic. This writer holds a value as
// an integer where it can prove that the value is a whole number, and writes the operation as integer or bit
// arithmetic. Every other operation is written by emit.ts's row writer, as loop-emit.ts writes it, with a conversion
// where a whole number meets an f32 statement. The program doesn't change, and each rule below is stated over what a
// compiled stage holds: an operation's kind, terms and weights, an index read's candidates and out-of-range rules, and
// the ranges, grids and exact bits that the circuit keeps per wire.
//
// What is proven. A value is "whole" when it is a whole number within a range inside ±2^24, e.g. a class within 0..16,
// and a whole number within 0..1 is a bit. A whole number is an i32 in the shader, and a bit is a u32 that holds 0 or 1.
// - A source cell is whole when its declaration says so: an exact bit (inputBit), or a grid of 1 or more with a range
//   inside ±2^24, e.g. input([0, 16], {grid: 1}).
// - An operation is whole when its operands are whole, its weights and biases are whole numbers, and the largest value
//   that any product or partial sum can take stays within 2^24. FP32 then computes every one of them without rounding,
//   in any order, so i32 arithmetic gives the same value. Each weight is itself within 2^24, since it is written as
//   an i32 literal. A boolean atom is a bit. An index read is whole when every candidate that its index can reach is
//   whole. An operation that the circuit declares on a grid with a small range is whole too, e.g. the bit that
//   emitCompare returns from a sign cascade.
// - A register or a field is held as an integer when the cells bound to it are declared whole and every stage that
//   writes it writes a proven whole number, a bit for a register of bits. The range of a cell that is bound to a
//   register comes from the cell's declaration, e.g. a match count within 0..512: the writer proves that every write is
//   a whole number, and doesn't prove that a counter stays below its declared maximum.
// - A register or a field that isn't proven is held as f32, and then its cell's declared grid isn't known to hold,
//   e.g. a level declared a whole number into which a stage writes level + 0.25. The circuit infers the grid of a sum
//   or a product from its operands' grids, so the circuit's grid for an operation is taken only when it holds for
//   every operand: 2 × level is no whole number there, whatever the circuit says.
//
// The rules, each with why the integer statement gives the f32 statement's value:
// - Sum. An affine operation over whole numbers is an i32 sum, e.g. `1i - i32(b) + 2i * i32(c)`. JS: !bit, as 1 - bit,
//   is `bit ^ 1u`.
// - Gate. A ReLU gate max(0, bias + Σ weight × value) whose largest possible value is 1, where every value with a
//   positive weight is a bit and every value with a negative weight is never negative, is 1 exactly when every bit with
//   a positive weight is 1 and every value with a negative weight is 0: if one bit with a positive weight is 0, or one
//   value with a negative weight is 1 or more, the sum is 0 or less. E.g. relu(a - b) is `a & (b ^ 1u)`, and
//   relu(1 - a - b - c) is `(a | b | c) ^ 1u`. Any other gate within 0..1 is `u32(sum > 0i)`, and a gate that can
//   exceed 1 is `max(sum, 0i)`.
// - Comparison. In a gate, a value with a negative weight that is itself max(0, sum) of whole numbers is 0 exactly when
//   sum <= 0, so the pair of gates that emitCompare builds for [codePoint >= boundary], relu(1 - relu(boundary -
//   codePoint)), is `u32(boundary <= codePoint)`.
// - Read. An index read whose index form is whole computes the index in i32, without a floor, and drops a clamp or a
//   select that the index's range rules out.
// - Packed reads. A gated read is an index read of bits whose index form is base + gate: `gate` sums the bits whose
//   weight is at least the table's length, and for every combination of those bits either gate is one fixed number, and
//   the read is table[base], or the index lies past the table's end, where the read is +0. So the read is
//   table[base] && open, with `open` a function of the gating bits. JS for regex search's
//   transitions[class][from][to] && bit: an index of class + 16 × (1 - bit) in a table of 16. When two or more gated
//   reads with a negative weight in one gate have the same base, the gate needs "none of table_k[base] && open_k", which
//   is `(word[base] & opens) == 0u`: bit k of word[base] is table_k[base], and bit k of `opens` is open_k. The words are
//   in the buffer packed_words, which the runner fills from the shared cells with packWords. Two bits in one gate form
//   are an OR only when they are never both 1, which the writer proves by taking both as 1 and following what each gate
//   then says about its own operands until one value would be both 0 and 1, e.g. a start bit that needs "no state
//   continues" next to a bit that says "state 0 continues". A read with more than 4 gating bits isn't taken as a gated
//   read and is read on its own, see maximumGatingBits.
// - Lanes. Bits that are stored together are one u32 word, each bit at its own position in the word, its lane: a field
//   of bits, and registers of bits that the same stages write, e.g. the 8 reached[s] bits. When the same gate or
//   negation is computed for every bit of such a list, one word operation computes all of them, since AND, OR and XOR
//   work on each bit of a word on its own, e.g. continuing[s] = relu(reached[s] - notReach[s]) for 8 states is
//   `reached & (notReach ^ 255u)`. "None of these bits is 1" over such a word is `word == 0u`. A list of gated reads
//   with the same base is the packed word itself, and a list of shared cells is packed into a word once per document.
//   Any other list of bits is written into its lanes one by one, `b0 | (b1 << 1u) | ...`, as loop-emit.ts writes a
//   field of bits. Where the order of a list is free, i.e. for the values that must all be 0 in one gate, the writer
//   takes the order of the stored lanes that the values are computed from, so a gate's reads can be built in any
//   order of the states. A word holds 32 bits, and a longer list is taken 32 bits at a time.
//
// Statements are written when a later statement or an output first needs them, so an operation that only feeds a packed
// read, a comparison or a word operation gets no statement of its own.
import {type Circuit, type Wire, invariant} from '../runtime/numeric-ui-core.ts'
import {type IndexRow, type Row, type Term, createRowWriter, flattenProgram, positionsOfWires, wgslFloat, wgslInt} from './emit.ts'
import type {LoopWgslModule} from './loop-emit.ts'
import type {Binding, LoopProgram, Stage} from './loop-program.ts'

// A whole number within minimum..maximum, both within ±2^24.
type Whole = {minimum: number, maximum: number}
type ReluRow = Extract<Row, {kind: 'reglu'}>
// How a register or a field is held: one u32 per bit, an i32, or an f32 as loop-emit.ts holds it.
type Storage = 'bit' | 'whole' | 'float'
// One value of a stage in the shader: the text that reads it, and the type of that text. A 'bit' is a u32 of 0 or 1.
type Held = {text: string, type: 'bit' | 'int' | 'float'}

const wholeLimit = 2 ** 24
const isBit = (whole: Whole) => whole.minimum >= 0 && whole.maximum <= 1

/** @public */
export type RuleCounts = {
  // Operations written as integer or bit statements, by rule.
  sum: number
  gate: number
  // The inner gates that became one `<=`.
  comparison: number
  // Other ReLU gates and products in i32.
  product: number
  read: number
  // Index reads that became one bit of a packed word, and how many words are read for them per position.
  packedRead: number
  packedWord: number
  // Gates and negations that one word operation computes for every lane of a word.
  lane: number
  // Operations written as loop-emit.ts writes them.
  float: number
}

type BitLoopWgslModule = LoopWgslModule & {
  // The plan of the buffer packed_words, for the runner: bit b of word w is the shared cell packedCells[w][b], which
  // holds 0 or 1. A word without cells is 0. Empty when the shader has no such buffer.
  packedCells: readonly (readonly number[])[]
  rules: RuleCounts
}

// The contents of packed_words for the given shared values, e.g. after a new pattern's cells were written.
export function packWords(packedCells: BitLoopWgslModule['packedCells'], shared: Float32Array, target = new Uint32Array(packedCells.length)) {
  invariant(target.length === packedCells.length, 'the target has one word per packed word')
  for (let word = 0; word < packedCells.length; word++) {
    const cells = packedCells[word]!
    let bits = 0
    for (let bit = 0; bit < cells.length; bit++) if (shared[cells[bit]!] !== 0) bits |= 1 << bit
    target[word] = bits
  }
  return target
}

// What a wire's own declarations prove: an exact bit, or a grid of whole numbers with a small range.
function declaredWhole(circuit: Circuit, wire: number): Whole | undefined {
  const range = circuit.ranges[wire]!
  if (circuit.exactBits[wire] === true) return {minimum: 0, maximum: 1}
  const grid = circuit.grids[wire]
  if (grid === undefined || !Number.isInteger(grid) || !(range.minimum >= -wholeLimit && range.maximum <= wholeLimit)) return undefined
  return narrowed({minimum: -wholeLimit, maximum: wholeLimit}, range)
}

// A whole number that also lies within a range that the circuit keeps for it, e.g. one declared with assumeRange.
function narrowed(whole: Whole, range: {minimum: number, maximum: number}): Whole | undefined {
  const minimum = Math.max(whole.minimum, Math.ceil(range.minimum))
  const maximum = Math.min(whole.maximum, Math.floor(range.maximum))
  // A range without a whole number in it contradicts the proof, and proves nothing.
  return minimum <= maximum ? {minimum, maximum} : undefined
}

// The range of bias + Σ weight × value when every value is whole and nothing can round, see the top of the file.
function sumRange(proofs: readonly (Whole | undefined)[], terms: readonly Term[], bias: number): Whole | undefined {
  if (!Number.isInteger(bias)) return undefined
  let minimum = bias
  let maximum = bias
  let magnitude = Math.abs(bias)
  for (const {weight, position} of terms) {
    const operand = proofs[position]
    // A weight is written as an i32 literal, so it has to be a whole number that i32 holds, also where its product is
    // small, e.g. a sign cascade's 2^127 times a value that is always 0.
    if (operand === undefined || !Number.isInteger(weight) || Math.abs(weight) > wholeLimit) return undefined
    minimum += weight > 0 ? weight * operand.minimum : weight * operand.maximum
    maximum += weight > 0 ? weight * operand.maximum : weight * operand.minimum
    magnitude += Math.abs(weight) * Math.max(Math.abs(operand.minimum), Math.abs(operand.maximum))
  }
  return magnitude <= wholeLimit ? {minimum, maximum} : undefined
}

type StageProof = {
  rows: Row[]
  // By packed position: the source cells, then the operations.
  proofs: (Whole | undefined)[]
  // By packed position: true for an operation whose own sums are whole and can't round, so that integer statements
  // give its value. For an index read, the sum is the index form.
  exact: boolean[]
}

// What a stage's proof starts from for one source cell: a whole number, undefined for any other number, and 'unproven'
// for a register or a field whose cell is declared on a grid or as a bit while the writer holds it as f32, e.g. a
// level declared a whole number into which a stage writes level + 0.25.
type SourceProof = Whole | 'unproven' | undefined

function proveStage(stage: Stage, sourceProof: (position: number) => SourceProof): StageProof {
  const {program, circuit} = stage
  const rows = flattenProgram(program)
  const proofs: (Whole | undefined)[] = []
  const exact: boolean[] = []
  // By packed position: whether the circuit's grid for the value can be taken. The circuit infers the grid of a sum or
  // a product from its operands' grids, so it holds when theirs do, and an 'unproven' cell's doesn't: for a level that
  // holds 0.25, the circuit still says that 2 × level is a whole number. A boolean atom is 0 or 1 whatever it reads,
  // and an index read's value is one of its candidates.
  const gridHolds: boolean[] = []
  for (let position = 0; position < program.sourceSize; position++) {
    const proof = sourceProof(position)
    proofs.push(proof === 'unproven' ? undefined : proof)
    gridHolds.push(proof !== 'unproven')
    exact.push(false)
  }
  for (const row of rows) {
    let inferred: Whole | undefined
    let rowExact = false
    let holds = true
    switch (row.kind) {
      case 'affine':
        inferred = sumRange(proofs, row.terms, row.bias)
        rowExact = inferred !== undefined
        holds = row.terms.every(term => gridHolds[term.position]!)
        break
      case 'reglu': {
        holds = row.terms.every(term => gridHolds[term.position]!) && row.gateTerms.every(term => gridHolds[term.position]!)
        const left = sumRange(proofs, row.terms, row.bias)
        const gate = sumRange(proofs, row.gateTerms, row.gateBias)
        if (left === undefined || gate === undefined) break
        const low = Math.max(0, gate.minimum)
        const high = Math.max(0, gate.maximum)
        const products = [left.minimum * low, left.minimum * high, left.maximum * low, left.maximum * high]
        inferred = {minimum: Math.min(...products), maximum: Math.max(...products)}
        if (Math.max(Math.abs(inferred.minimum), Math.abs(inferred.maximum)) > wholeLimit) inferred = undefined
        rowExact = inferred !== undefined
        break
      }
      case 'boolean':
        inferred = {minimum: 0, maximum: 1}
        break
      case 'index': {
        const index = sumRange(proofs, row.indexTerms, row.indexBias)
        rowExact = index !== undefined
        const last = row.candidateCount - 1
        // The candidates that the index can reach, and +0 from a side that reads zero.
        const first = index === undefined ? 0 : Math.min(last, Math.max(0, index.minimum))
        const final = index === undefined ? last : Math.min(last, Math.max(0, index.maximum))
        const zero = (row.belowZero && (index === undefined || index.minimum < 0)) || (row.aboveZero && (index === undefined || index.maximum > last))
        inferred = zero ? {minimum: 0, maximum: 0} : {minimum: Number.POSITIVE_INFINITY, maximum: Number.NEGATIVE_INFINITY}
        for (let candidate = first; candidate <= final; candidate++) {
          const whole = proofs[row.candidateBase + candidate * row.candidateStride]
          if (whole === undefined) {
            inferred = undefined
            break
          }
          inferred = {minimum: Math.min(inferred.minimum, whole.minimum), maximum: Math.max(inferred.maximum, whole.maximum)}
        }
        break
      }
    }
    const wire = program.wireOrder[row.position]!
    // An operation that the range arithmetic above doesn't reach is whole when the circuit says so, e.g. the bit that
    // emitCompare declares at the end of a sign cascade.
    const proof = inferred !== undefined ? narrowed(inferred, circuit.ranges[wire]!) : holds ? declaredWhole(circuit, wire) : undefined
    proofs.push(proof)
    gridHolds.push(holds)
    // An operation whose proof the circuit's own range contradicts is written as loop-emit.ts writes it. An index read
    // keeps its integer index when the value that it reads isn't whole.
    exact.push(rowExact && (row.kind === 'index' || proof !== undefined))
  }
  return {rows, proofs, exact}
}

// How a register or a field can be held, from the declarations of the cells bound to it: bits when every such cell is
// a bit, whole numbers when every one is whole, and f32 values otherwise, or when no stage reads it.
function declaredStorage(wholes: readonly (Whole | undefined)[]): Storage {
  if (wholes.length === 0 || wholes.includes(undefined)) return 'float'
  return wholes.every(whole => isBit(whole!)) ? 'bit' : 'whole'
}

// Programs of FP32 stages only: a word stage (runtime/words.ts) is integer code already, and loop-emit.ts writes it.
export function emitBitLoopWgsl(loop: LoopProgram, options: {workgroupSize?: number} = {}): BitLoopWgslModule {
  const workgroupSize = options.workgroupSize ?? 64
  const {positions} = loop
  for (const field of loop.fields) invariant(!field.bits || field.width <= 32, 'a field of bits holds at most 32 bits per position')

  // How each register and field is held. It starts from the declarations of the cells bound to it, and a register or
  // field that some stage writes a value into that isn't proven to fit is held as f32 instead, until nothing changes:
  // the stages are proven again, since a cell bound to an f32 register proves nothing, and neither does the circuit's
  // grid for an operation that reads the cell.
  const boundWholes = (isBound: (binding: Binding) => boolean) => loop.stages.flatMap(stage =>
    [...stage.bindings].filter(([, binding]) => isBound(binding)).map(([wire]) => declaredWhole(stage.circuit, wire)))
  const registerStorage = loop.registers.map(({initial}, index): Storage => {
    const storage = declaredStorage(boundWholes(binding => binding.kind === 'register' && binding.register === index))
    if (storage === 'bit' && (initial === 0 || initial === 1)) return 'bit'
    return storage !== 'float' && Number.isInteger(initial) && Math.abs(initial) <= wholeLimit ? 'whole' : 'float'
  })
  const fieldStorage = loop.fields.map((_, index) => declaredStorage(boundWholes(binding => binding.kind === 'field' && binding.field === index)))
  let stageProofs: StageProof[]
  for (;;) {
    stageProofs = loop.stages.map(stage => proveStage(stage, position => {
      const binding = stage.bindings.get(position as Wire)
      const storage = binding?.kind === 'register' ? registerStorage[binding.register]! : binding?.kind === 'field' ? fieldStorage[binding.field]! : undefined
      if (storage !== 'float') return declaredWhole(stage.circuit, position)
      return stage.circuit.grids[position] !== undefined || stage.circuit.exactBits[position] === true ? 'unproven' : undefined
    }))
    let changed = false
    for (const [stageIndex, stage] of loop.stages.entries()) {
      const outputPositions = positionsOfWires(stage.program, stage.outputs.map(output => output.wire))
      for (const [index, {write}] of stage.outputs.entries()) {
        if (write.kind !== 'register' && write.kind !== 'field') continue
        const storages = write.kind === 'register' ? registerStorage : fieldStorage
        const at = write.kind === 'register' ? write.register : write.field
        const proof = stageProofs[stageIndex]!.proofs[outputPositions[index]!]
        const fits = storages[at] === 'float' || (proof !== undefined && (storages[at] === 'whole' || isBit(proof)))
        if (!fits) {
          storages[at] = 'float'
          changed = true
        }
      }
    }
    if (!changed) break
  }
  // A field of bits is one u32 word per position, as in loop-emit.ts, when the loop program says so. A field that the
  // loop program declares as plain numbers is one too when its bits are proven and fit a word, and every stage that
  // writes the field writes all of a position's bits, e.g. not when two loops fill two of four bits each.
  const writesWholePositions = (field: number) => loop.stages.every(stage => {
    const components = new Set(stage.outputs.flatMap(({write}) => write.kind === 'field' && write.field === field ? [write.component] : []))
    return components.size === 0 || components.size === loop.fields[field]!.width
  })
  const fieldWords = loop.fields.map((field, index) => field.bits || (fieldStorage[index] === 'bit' && field.width <= 32 && writesWholePositions(index)))

  const bufferName = (buffer: number) => buffer === 0 ? 'instance_sources' : `instance_inputs_${buffer}`
  const baseName = (buffer: number) => buffer === 0 ? 'source_base' : `input_base_${buffer}`

  // Registers of bits can be the lanes of one u32 word, 32 a word, in the registers' order, e.g. the 8 reached[s] bits
  // that the forward step writes, so that one word operation reads or writes all of them. A word only pays when the
  // stages read it as a word, and a loop program doesn't say which registers it uses as one list. So the stages are
  // written with a first choice of words, and a word of which half the registers or more were read one by one, each
  // read a shift and an AND, is replaced and the stages are written again:
  // - first, the registers that the same stages write are one word
  // - then, of those, the registers that each stage writes with one kind of operation, e.g. 8 switches that a stage
  //   writes with gates, next to 8 "was on" bits that the same stage writes with negations
  // - then the registers are variables of their own.
  // What the stages write into a register: per stage, 'written', or with `form` the operation's kind and how many
  // of its weights are positive, e.g. 'reglu 1 3' for on && !blocked && !stuck.
  const writesOf = (register: number, form: boolean) => loop.stages.map((stage, stageIndex) => {
    const output = stage.outputs.find(({write}) => write.kind === 'register' && write.register === register)
    if (output === undefined) return ''
    if (!form) return 'written'
    const row = stageProofs[stageIndex]!.rows[positionsOfWires(stage.program, [output.wire])[0]! - stage.program.sourceSize]!
    const terms = row.kind === 'index' ? row.indexTerms : row.kind === 'reglu' ? [...row.gateTerms, ...row.terms] : row.terms
    return `${row.kind} ${terms.filter(term => term.weight > 0).length} ${terms.length}`
  }).join(',')
  // The registers with the same key, 32 a word, without a register that is alone.
  const wordsBy = (registers: readonly number[], keyOf: (register: number) => string) => {
    const groups = new Map<string, number[]>()
    for (const register of registers) {
      const group = groups.get(keyOf(register))
      if (group === undefined) groups.set(keyOf(register), [register])
      else group.push(register)
    }
    const words: number[][] = []
    for (const group of groups.values()) for (let from = 0; from < group.length; from += 32) words.push(group.slice(from, from + 32))
    return words.filter(word => word.length > 1)
  }
  // `split` marks a word of the second choice.
  let wordChoices = wordsBy(registerStorage.flatMap((storage, register) => storage === 'bit' ? [register] : []), register => writesOf(register, false)).map(registers => ({registers, split: false}))
  let written = writeStages(wordChoices.map(word => word.registers))
  for (;;) {
    const {laneReads} = written
    const pays = (word: {registers: number[]}, index: number) => laneReads[index]! * 2 < word.registers.length
    if (wordChoices.every(pays)) break
    wordChoices = wordChoices.flatMap((word, index) => {
      if (pays(word, index)) return [word]
      if (word.split) return []
      // A word whose registers are all written by one kind of operation has no smaller words to try.
      return wordsBy(word.registers, register => writesOf(register, true)).filter(registers => registers.length < word.registers.length).map(registers => ({registers, split: true}))
    })
    written = writeStages(wordChoices.map(word => word.registers))
  }
  const registerWords = wordChoices.map(word => word.registers)
  const {hoisted, packedCells, rules, rowCounts, stageLines} = written
  const laneOfRegister = (register: number) => {
    const word = registerWords.findIndex(registers => registers.includes(register))
    return word === -1 ? undefined : {word, lane: registerWords[word]!.indexOf(register)}
  }
  // Writes every stage's statements, with the given groups of registers as the lanes of words.
  function writeStages(words: readonly (readonly number[])[]) {
    const laneOf = new Map<number, {word: number, lane: number}>()
    words.forEach((registers, word) => registers.forEach((register, lane) => laneOf.set(register, {word, lane})))
    // How many times a stage read one register of each word on its own.
    const laneReads = words.map(() => 0)
    // The `let`s before the first stage, by name: shared cells, a document's cells at fixed columns, and words of
    // shared cells.
    const hoisted = new Map<string, string>()
    const sharedWords = new Map<string, string>()
    const packedCells: number[][] = []
    const packedTables = new Map<string, number>()
    const rules: RuleCounts = {sum: 0, gate: 0, comparison: 0, product: 0, read: 0, packedRead: 0, packedWord: 0, lane: 0, float: 0}
    const rowCounts = {affine: 0, reglu: 0, boolean: 0, index: 0}
    const stageLines: string[] = []

    loop.stages.forEach((stage, stageIndex) => {
      const {program} = stage
      const {sourceSize} = program
      const {rows, proofs, exact} = stageProofs[stageIndex]!
      for (const row of rows) rowCounts[row.kind]++
      const prefix = `${String.fromCharCode(97 + stageIndex)}_`
      const rowAt = (position: number) => position < sourceSize ? undefined : rows[position - sourceSize]!
      const held = new Map<number, Held>()
      const wordsRead = new Set<number>()
      // The f32 `let` of an operation that is held as an integer, for an f32 statement that reads it.
      const floatLets = new Set<number>()
      // Sums and words that several statements of this stage read, by their text, e.g. the word of a gate's open bits.
      const sharedLets = new Map<string, string>()
      const excluded = new Map<string, boolean>()
      const candidateText = (row: IndexRow, index: string) => {
        for (let candidate = 0; candidate < row.candidateCount; candidate++) {
          invariant(!stage.bindings.has((row.candidateBase + candidate * row.candidateStride) as Wire), 'in a loop program, an index read\'s candidates are shared cells')
        }
        const first = stage.sharedOffset + row.candidateBase
        return `shared_values[u32(${first}i + ${index}${row.candidateStride === 1 ? '' : ` * ${wgslInt(row.candidateStride)}`})]`
      }
      const writer = createRowWriter(sourceSize, {
        prefix,
        // writeFloat holds every operand before the row writer asks for it.
        source: position => floatText(position),
        candidate: candidateText,
      })
      const {body} = writer

      // The value at a position, loaded or written the first time something needs it.
      function hold(position: number): Held {
        let value = held.get(position)
        if (value === undefined) {
          const row = rowAt(position)
          value = row === undefined ? loadSource(position) : writeRow(row)
          held.set(position, value)
        }
        return value
      }
      const bitText = (position: number) => {
        const value = hold(position)
        invariant(proofs[position] !== undefined && isBit(proofs[position]), 'only a proven bit is read as a bit')
        return value.type === 'bit' ? value.text : `u32(${value.text})`
      }
      const intText = (position: number) => {
        const value = hold(position)
        invariant(proofs[position] !== undefined, 'only a proven whole number is read as an integer')
        return value.type === 'int' ? value.text : `i32(${value.text})`
      }
      function floatText(position: number) {
        const value = hold(position)
        if (value.type === 'float') return value.text
        if (position < sourceSize) return `f32(${value.text})`
        // The row writer names an operation's f32 value itself, e.g. b_v150.
        const name = `${prefix}v${position}`
        if (!floatLets.has(position)) {
          floatLets.add(position)
          body.push(`let ${name}: f32 = f32(${value.text});`)
        }
        return name
      }

      // The word that holds this position's bits of a field, read once per position.
      function fieldWord(field: number) {
        const word = `${prefix}w${field}`
        if (!wordsRead.has(field)) {
          wordsRead.add(field)
          body.push(`let ${word}: u32 = field${field}[j];`)
        }
        return word
      }

      function loadSource(position: number): Held {
        const proof = proofs[position]
        const type = proof === undefined ? 'float' : isBit(proof) ? 'bit' : 'int'
        // A cell that is read from an f32 buffer, as a `let` of the proven type, e.g. `let s131b: u32 = u32(shared_values[131u]);`.
        const converted = (name: string, read: string, lines: {push: (line: string) => void} | undefined) => {
          const text = type === 'float' ? name : `${name}${type === 'bit' ? 'b' : 'i'}`
          const line = type === 'float' ? `let ${text}: f32 = ${read};` : `let ${text}: ${type === 'bit' ? 'u32' : 'i32'} = ${type === 'bit' ? 'u32' : 'i32'}(${read});`
          if (lines === undefined) hoisted.set(text, line)
          else lines.push(line)
          return {text, type} as const
        }
        const binding = stage.bindings.get(position as Wire)
        if (binding === undefined) return converted(`s${stage.sharedOffset + position}`, `shared_values[${stage.sharedOffset + position}u]`, undefined)
        switch (binding.kind) {
          case 'cell': return converted(`d${binding.buffer}_${binding.column}`, `${bufferName(binding.buffer)}[${baseName(binding.buffer)} + ${binding.column}u]`, undefined)
          case 'item': return converted(`${prefix}c${position}`, `${bufferName(binding.buffer)}[${baseName(binding.buffer)} + j]`, body)
          case 'register': {
            const storage = registerStorage[binding.register]!
            const lane = laneOf.get(binding.register)
            if (lane !== undefined) {
              laneReads[lane.word]!++
              return {text: `((rw${lane.word} >> ${lane.lane}u) & 1u)`, type: 'bit'}
            }
            return {text: `r${binding.register}`, type: storage === 'bit' ? 'bit' : storage === 'whole' ? 'int' : 'float'}
          }
          case 'field': {
            const field = loop.fields[binding.field]!
            const storage = fieldStorage[binding.field]!
            const name = `${prefix}f${position}`
            if (fieldWords[binding.field]!) {
              const bit = `(${fieldWord(binding.field)} >> ${binding.component}u) & 1u`
              if (storage === 'bit') {
                body.push(`let ${name}b: u32 = ${bit};`)
                return {text: `${name}b`, type: 'bit'}
              }
              body.push(`let ${name}: f32 = f32(${bit});`)
              return {text: name, type: 'float'}
            }
            const read = `field${binding.field}[j * ${field.width}u + ${binding.component}u]`
            if (storage === 'float') {
              body.push(`let ${name}: f32 = ${read};`)
              return {text: name, type: 'float'}
            }
            body.push(`let ${name}i: i32 = ${read};`)
            return {text: `${name}i`, type: 'int'}
          }
        }
      }

      // bias + Σ weight × value as an i32 expression, e.g. `16i * i32(r3) + a_v208i`.
      function intSum(terms: readonly Term[], bias: number) {
        let text = ''
        for (const {weight, position} of terms) {
          const magnitude = Math.abs(weight)
          const product = magnitude === 1 ? intText(position) : `${magnitude}i * ${intText(position)}`
          if (text === '') text = weight < 0 ? `-${product}` : product
          else text = `${text} ${weight < 0 ? '-' : '+'} ${product}`
        }
        if (text === '') return wgslInt(bias)
        return bias === 0 ? text : `${text} ${bias < 0 ? '-' : '+'} ${Math.abs(bias)}i`
      }
      // A sum as one factor of a product.
      const grouped = (sum: string) => sum.includes(' ') ? `(${sum})` : sum
      // Bits written into the lanes of a word one by one, bit k into lane k.
      const inLanes = (bits: readonly string[]) => bits.map((bit, lane) => lane === 0 ? bit : `(${bit} << ${lane}u)`).join(' | ')
      // A sum as one operand of a larger expression: a name as it is, anything else as a `let` that later statements
      // with the same sum read too.
      function sharedLet(type: 'i32' | 'u32', text: string) {
        if (/^\w+$/.test(text)) return text
        let name = sharedLets.get(text)
        if (name === undefined) {
          name = `${prefix}${type === 'i32' ? 'x' : 'm'}${sharedLets.size}`
          sharedLets.set(text, name)
          body.push(`let ${name}: ${type} = ${text};`)
        }
        return name
      }
      // `sum <= 0` for a sum of whole numbers, with the terms that are subtracted on the right, e.g. `s1i <= a_c0i` for
      // boundary - codePoint.
      function atMostZero(terms: readonly Term[], bias: number) {
        const added = terms.filter(term => term.weight > 0)
        const subtracted = terms.filter(term => term.weight < 0).map(term => ({weight: -term.weight, position: term.position}))
        return `${intSum(added, Math.max(bias, 0))} <= ${intSum(subtracted, Math.max(-bias, 0))}`
      }

      // JS: !bit, as 1 - bit. The bit's position, or undefined for any other operation.
      const negatedBit = (row: Row) => row.kind === 'affine' && exact[row.position]! && row.bias === 1 && row.terms.length === 1 && row.terms[0]!.weight === -1 && isBit(proofs[row.terms[0]!.position]!)
        ? row.terms[0]!.position
        : undefined
      // The Gate rule's two lists for max(0, bias + Σ weight × value): the bits that must all be 1 and the values that
      // must all be 0 for the gate to be 1, or undefined when the gate isn't of that form.
      function conjunction(row: ReluRow): {all: number[], none: number[]} | undefined {
        if (!exact[row.position]! || row.terms.length > 0 || row.bias !== 1) return undefined
        let largest = row.gateBias
        const all: number[] = []
        const none: number[] = []
        for (const {weight, position} of row.gateTerms) {
          const operand = proofs[position]!
          if (weight > 0) {
            if (!isBit(operand)) return undefined
            all.push(position)
            largest += weight
          } else {
            if (operand.minimum < 0) return undefined
            none.push(position)
          }
        }
        return largest === 1 ? {all, none} : undefined
      }

      // Whether two bits are never both 1. Both are taken as 1, and each gate that is then 1 says what its own operands
      // are, until one value would have to be both 0 and 1, e.g. `start` needs notContinuing, which needs every
      // continuing[s] to be 0, next to continuing[0] taken as 1.
      function neverBothOne(first: number, second: number) {
        const key = `${Math.min(first, second)},${Math.max(first, second)}`
        let proven = excluded.get(key)
        if (proven === undefined) {
          proven = false
          const forced = new Map<number, boolean>()
          const pending: [position: number, one: boolean][] = [[first, true], [second, true]]
          while (pending.length > 0 && !proven) {
            const [position, one] = pending.pop()!
            const known = forced.get(position)
            if (known !== undefined) {
              proven = known !== one
              continue
            }
            forced.set(position, one)
            const row = rowAt(position)
            if (row === undefined) continue
            const negated = negatedBit(row)
            if (negated !== undefined) {
              pending.push([negated, !one])
            } else if (row.kind === 'reglu' && one) {
              const gate = conjunction(row)
              if (gate === undefined) continue
              for (const operand of gate.all) pending.push([operand, true])
              for (const operand of gate.none) pending.push([operand, false])
            }
          }
          excluded.set(key, proven)
        }
        return proven
      }

      // The Packed reads rule's reading of one index read, or undefined when the read isn't a gated read of bits.
      type Gated = {
        // The index that the read has when it is open, a whole number within `range`.
        baseTerms: Term[]
        baseBias: number
        range: Whole
        // The bits that decide whether the read is open, and for each combination of them, bit 0 first, whether it is.
        gating: number[]
        open: number
        // Two reads with the same key read their tables at the same index, under the same out-of-range rules.
        key: string
      }
      // A read with more gating bits than this isn't taken as a gated read, and is read on its own, as the Read rule
      // writes it, e.g. a read of table[row - 5 × (wanted + a + b + c + d)], which has five. Two reasons. The rule
      // goes through every combination of the gating bits, 16 for 4 bits. And `open` holds one bit per combination,
      // set with `1 << combination` and written into the shader as a u32 literal: with 5 bits that is 32 combinations,
      // and JavaScript's `1 << 31` is negative, so a read that is open with all five bits at 1 would get the literal
      // `-2147483648u`, which isn't WGSL.
      const maximumGatingBits = 4
      function gatedRead(row: IndexRow): Gated | undefined {
        if (!exact[row.position]!) return undefined
        // The words are packed from every candidate, also from one that the index never reaches, so each one has to be
        // a shared cell that is declared to hold 0 or 1: an exact bit, a cell on a grid of 1 within 0..1, or a constant.
        for (let candidate = 0; candidate < row.candidateCount; candidate++) {
          const cell = row.candidateBase + candidate * row.candidateStride
          if (stage.bindings.has(cell as Wire) || proofs[cell] === undefined || !isBit(proofs[cell])) return undefined
        }
        const isGating = (term: Term) => isBit(proofs[term.position]!) && Math.abs(term.weight) >= row.candidateCount
        const gatingTerms = row.indexTerms.filter(isGating)
        const baseTerms = row.indexTerms.filter(term => !isGating(term))
        const base = sumRange(proofs, baseTerms, 0)
        if (gatingTerms.length > maximumGatingBits || base === undefined) return undefined
        const last = row.candidateCount - 1
        let open = 0
        let baseBias: number | undefined
        for (let combination = 0; combination < 2 ** gatingTerms.length; combination++) {
          let gate = row.indexBias
          let possible = true
          for (const [bit, term] of gatingTerms.entries()) {
            if (((combination >> bit) & 1) === 0) continue
            gate += term.weight
            for (let earlier = 0; earlier < bit; earlier++) {
              if (((combination >> earlier) & 1) === 1 && neverBothOne(gatingTerms[earlier]!.position, term.position)) possible = false
            }
          }
          // A combination that never happens can count as open, which keeps e.g. an OR of two bits an OR.
          const closed = (row.aboveZero && gate + base.minimum > last) || (row.belowZero && gate + base.maximum < 0)
          if (!possible || closed) {
            if (!possible) open |= 1 << combination
            continue
          }
          if (baseBias !== undefined && baseBias !== gate) return undefined
          baseBias = gate
          open |= 1 << combination
        }
        if (baseBias === undefined) return undefined
        return {
          baseTerms, baseBias, range: {minimum: base.minimum + baseBias, maximum: base.maximum + baseBias}, gating: gatingTerms.map(term => term.position), open,
          key: `${row.candidateCount},${row.belowZero},${row.aboveZero},${baseBias},${baseTerms.map(term => `${term.position}*${term.weight}`).join(',')}`,
        }
      }
      // `open` as a u32 expression of 0 or 1 over the gating bits.
      function openText({gating, open}: Gated) {
        const combinations = 2 ** gating.length
        const every = 2 ** combinations - 1
        if (open === every) return '1u'
        const bits = gating.map(bitText)
        if (gating.length === 1) return open === 2 ? bits[0]! : `(${bits[0]!} ^ 1u)`
        if (open === every - 1) return `(${bits.join(' | ')})`
        return `((${open}u >> (${bits.map((bit, index) => index === 0 ? bit : `(${bit} << ${index}u)`).join(' | ')})) & 1u)`
      }
      // The word whose bit k is gated read k, for reads that share a key: one read of a packed word, and one AND with
      // the word of the reads' open bits.
      function packedWord(gated: Gated, reads: readonly IndexRow[]) {
        const first = reads[0]!
        const last = first.candidateCount - 1
        // A word of zeros before or after the table stands for a side that reads +0.
        const zeroBefore = first.belowZero && gated.range.minimum < 0
        const zeroAfter = first.aboveZero && gated.range.maximum > last
        const tableKey = `${stage.sharedOffset},${zeroBefore},${zeroAfter},${first.candidateCount},${reads.map(read => `${read.candidateBase}:${read.candidateStride}`).join(',')}`
        let start = packedTables.get(tableKey)
        if (start === undefined) {
          start = packedCells.length
          packedTables.set(tableKey, start)
          if (zeroBefore) packedCells.push([])
          for (let candidate = 0; candidate <= last; candidate++) packedCells.push(reads.map(read => stage.sharedOffset + read.candidateBase + candidate * read.candidateStride))
          if (zeroAfter) packedCells.push([])
        }
        const lowest = zeroBefore ? -1 : 0
        const highest = zeroAfter ? last + 1 : last
        let index = sharedLet('i32', intSum(gated.baseTerms, gated.baseBias))
        if (gated.range.minimum < lowest || gated.range.maximum > highest) index = `clamp(${index}, ${wgslInt(lowest)}, ${wgslInt(highest)})`
        rules.packedRead += reads.length
        rules.packedWord++
        const word = `packed_words[u32(${start - lowest === 0 ? '' : `${start - lowest}i + `}${index})]`
        const opens = opensOf(reads)
        const firstBits = opens.gateds.map(({gating}) => gating[0]!)
        switch (opens.kind) {
          case 'always': return word
          case 'not': return `${word} & ${sharedLet('u32', `${wordOfBits(firstBits)} ^ ${lanesMask(reads.length)}`)}`
          case 'any': {
            const further = opens.gateds.flatMap(({gating}, lane) => gating.slice(1).map(position => lane === 0 ? bitText(position) : `(${bitText(position)} << ${lane}u)`))
            return `${word} & ${sharedLet('u32', [wordOfBits(firstBits), ...further].join(' | '))}`
          }
          case 'each': return `${word} & ${sharedLet('u32', inLanes(opens.gateds.map(openText)))}`
        }
      }
      // How the word of some gated reads' open bits is built: no word when every read is always open ('always'), the
      // gating bits as a word, negated ('not': each read is open exactly when its one gating bit is 0) or with each
      // read's further bits in its lane ('any': each read is open when one of its gating bits is 1), or every open bit
      // written into its lane ('each').
      function opensOf(reads: readonly IndexRow[]) {
        const gateds = reads.map(read => gatedRead(read)!)
        const every = (gating: readonly number[]) => 2 ** (2 ** gating.length) - 1
        const kind = gateds.every(({gating, open}) => open === every(gating)) ? 'always' as const
          : gateds.every(({gating, open}) => gating.length === 1 && open === 1) ? 'not' as const
            : gateds.every(({gating, open}) => gating.length >= 1 && open === every(gating) - 1) ? 'any' as const : 'each' as const
        return {kind, gateds}
      }

      // How a gate tests that one of its values with a negative weight is 0: a gated read goes into a packed word, a
      // comparison's inner gate becomes `<=`, a bit is tested as a bit, and another whole number with `== 0i`.
      function zeroTest(position: number): 'read' | 'comparison' | 'bit' | 'number' {
        const row = rowAt(position)
        if (row?.kind === 'index' && isBit(proofs[position]!) && gatedRead(row) !== undefined) return 'read'
        if (row?.kind === 'reglu' && exact[position]! && row.terms.length === 0 && row.bias === 1 && conjunction(row) === undefined) return 'comparison'
        return isBit(proofs[position]!) ? 'bit' : 'number'
      }

      // The Lanes rule. A list of bits as one u32 word, bit k of the word being the bit at positions[k]:
      //   'stored'  the bits are neighboring lanes of one field's word or of one word of registers
      //   'shared'  the bits are shared cells, packed into a word once per document
      //   'reads'   the bits are gated reads with the same base, so the word is the Packed reads rule's word
      //   'not'     every bit is the negation of a bit, so the word is the operands' word with every lane flipped
      //   'gate'    every bit is a gate of the Gate rule, each with as many bits that must be 1 and values that must
      //             be 0 as the others, so the word is the AND of the operands' words, e.g. reached & ~notReach for
      //             the 8 continuing[s] bits
      // A list that is none of these is built lane by lane, `b0 | (b1 << 1u) | ...`, which is the 'lanes' slot.
      type WordPlan =
        | {kind: 'stored', positions: readonly number[], word: () => string, first: number, width: number}
        | {kind: 'shared', positions: readonly number[]}
        | {kind: 'reads', positions: readonly number[], gated: Gated, reads: IndexRow[]}
        | {kind: 'not', positions: readonly number[], operand: WordSlot}
        | {kind: 'gate', positions: readonly number[], all: WordSlot[], none: WordSlot[]}
      type WordSlot = WordPlan | {kind: 'lanes', positions: readonly number[]}
      // The stored word and the lane of a source cell that is one bit of a field's word or of a word of registers.
      function storedLane(position: number) {
        const binding = stage.bindings.get(position as Wire)
        if (binding?.kind === 'register') {
          const lane = laneOf.get(binding.register)
          return lane === undefined ? undefined : {key: `r${lane.word}`, lane: lane.lane, word: () => `rw${lane.word}`, width: words[lane.word]!.length}
        }
        const {field} = binding?.kind === 'field' ? binding : {field: -1}
        if (binding?.kind !== 'field' || !fieldWords[field]! || fieldStorage[field] !== 'bit') return undefined
        return {key: `f${field}`, lane: binding.component, word: () => fieldWord(field), width: loop.fields[field]!.width}
      }
      // The stored lane that a bit is computed from: its own, or that of the first operand that has one, through
      // negations, gates of the Gate rule and the gating bits of gated reads. E.g. reached[3]'s lane for
      // continuing[3] = reached[3] && !notReach[3], and for a read that is gated by continuing[3].
      const origins = new Map<number, {key: string, lane: number} | undefined>()
      function originOf(position: number): {key: string, lane: number} | undefined {
        if (origins.has(position)) return origins.get(position)
        const row = rowAt(position)
        let operands: readonly number[] = []
        if (row !== undefined) {
          const negated = negatedBit(row)
          const gate = row.kind === 'reglu' ? conjunction(row) : undefined
          if (negated !== undefined) operands = [negated]
          else if (gate !== undefined) operands = [...gate.all, ...gate.none]
          else if (row.kind === 'index') operands = gatedRead(row)?.gating ?? []
        }
        let origin: {key: string, lane: number} | undefined = row === undefined ? storedLane(position) : undefined
        for (let index = 0; index < operands.length && origin === undefined; index++) origin = originOf(operands[index]!)
        origins.set(position, origin)
        return origin
      }
      // A list of bits whose order is free, e.g. the values that must all be 0 in a gate, as lists of at most 32 bits
      // that can each be one word: the bits that are computed from one stored word, in the order of its lanes, and the
      // other bits in the order given. So a gate's reads can be built in any order of the states, e.g. state 7 first,
      // and their gating bits are still the stored word.
      function wordLists<Item>(items: readonly Item[], positionOf: (item: Item) => number): Item[][] {
        const byWord = new Map<string, {item: Item, lane: number}[]>()
        for (const item of items) {
          const origin = originOf(positionOf(item))
          const key = origin?.key ?? ''
          const entries = byWord.get(key)
          if (entries === undefined) byWord.set(key, [{item, lane: origin?.lane ?? 0}])
          else entries.push({item, lane: origin?.lane ?? 0})
        }
        const lists: Item[][] = []
        for (const entries of byWord.values()) {
          // The sort keeps the given order of two bits from one lane, and of the bits without a stored word.
          entries.sort((left, right) => left.lane - right.lane)
          for (let from = 0; from < entries.length; from += 32) lists.push(entries.slice(from, from + 32).map(entry => entry.item))
        }
        return lists
      }
      function wordPlan(positions: readonly number[]): WordPlan | undefined {
        invariant(positions.length <= 32, 'a word holds at most 32 bits')
        if (positions.length < 2 || !positions.every(position => proofs[position] !== undefined && isBit(proofs[position]))) return undefined
        const slot = (operands: readonly number[]): WordSlot => wordPlan(operands) ?? {kind: 'lanes', positions: operands}
        const lanes = positions.map(storedLane)
        const first = lanes[0]
        if (first !== undefined && lanes.every((lane, index) => lane?.key === first.key && lane.lane === first.lane + index)) return {kind: 'stored', positions, word: first.word, first: first.lane, width: first.width}
        if (positions.every(position => position < sourceSize && !stage.bindings.has(position as Wire))) return {kind: 'shared', positions}
        const rowsOf = positions.map(rowAt)
        const reads = rowsOf.filter(row => row?.kind === 'index')
        const gateds = reads.map(gatedRead).filter(gated => gated !== undefined)
        if (gateds.length === positions.length && gateds.every(gated => gated.key === gateds[0]!.key)) return {kind: 'reads', positions, gated: gateds[0]!, reads}
        const negated = rowsOf.flatMap(row => row === undefined ? [] : [negatedBit(row)]).filter(operand => operand !== undefined)
        if (negated.length === positions.length) return {kind: 'not', positions, operand: slot(negated)}
        const gates = rowsOf.flatMap(row => row?.kind === 'reglu' ? [conjunction(row)] : []).filter(gate => gate !== undefined)
        const shape = gates[0]
        if (shape === undefined || gates.length !== positions.length) return undefined
        // A comparison or a number among the values that must be 0 is tested on its own, so such gates stay one
        // statement each.
        if (!gates.every(gate => gate.all.length === shape.all.length && gate.none.length === shape.none.length && gate.none.every(operand => zeroTest(operand) === 'bit' || zeroTest(operand) === 'read'))) return undefined
        const none = shape.none.map((_, index) => slot(gates.map(gate => gate.none[index]!)))
        // The values that must be 0 are taken lane by lane, the first value of every gate, the second, and so on, when
        // each of those lists is a word without building it lane by lane. When one isn't, a gate that tests its own
        // values as a word keeps them, e.g. each notReach[s] gate its packed word of 8 reads: taken lane by lane, the
        // open bit of every lane would be the same register, written 8 times.
        if (!none.every(isWhole) && gates.some(gate => testsWords(zeroTests(gate.none)))) return undefined
        return {kind: 'gate', positions, all: shape.all.map((_, index) => slot(gates.map(gate => gate.all[index]!))), none}
      }
      const isWhole = (slot: WordSlot): boolean => {
        switch (slot.kind) {
          case 'lanes': return false
          case 'stored': case 'shared': return true
          case 'reads': {
            // The reads' open bits must be a whole word too.
            const opens = opensOf(slot.reads)
            if (opens.kind === 'always') return true
            if (opens.kind === 'each' || opens.gateds.some(each => each.gating.length > 1)) return false
            const gating = wordPlan(opens.gateds.map(each => each.gating[0]!))
            return gating !== undefined && isWhole(gating)
          }
          case 'not': return isWhole(slot.operand)
          case 'gate': return slot.all.every(isWhole) && slot.none.every(isWhole)
        }
      }
      const lanesMask = (count: number) => `${count === 32 ? 0xffffffff : 2 ** count - 1}u`
      // The word's text, the same for the same list of bits.
      const wordTexts = new Map<string, string>()
      function wordOf(slot: WordSlot): string {
        const {positions} = slot
        const key = positions.join(',')
        let text = wordTexts.get(key)
        if (text !== undefined) return text
        const mask = lanesMask(positions.length)
        switch (slot.kind) {
          case 'lanes':
            text = sharedLet('u32', inLanes(positions.map(bitText)))
            break
          case 'stored': {
            const shifted = slot.first === 0 ? slot.word() : `(${slot.word()} >> ${slot.first}u)`
            // A stored word holds 0 above its last lane.
            text = slot.first + positions.length === slot.width ? shifted : `(${shifted} & ${mask})`
            break
          }
          case 'shared': {
            // One `let` before the first stage, which every stage that reads the same cells as a word shares.
            const cells = positions.map(position => stage.sharedOffset + position).join(',')
            text = sharedWords.get(cells)
            if (text === undefined) {
              text = `sw${sharedWords.size}`
              sharedWords.set(cells, text)
              hoisted.set(text, `let ${text}: u32 = ${inLanes(positions.map(bitText))};`)
            }
            break
          }
          case 'reads':
            text = sharedLet('u32', packedWord(slot.gated, slot.reads))
            break
          case 'not':
            rules.lane += positions.length
            text = sharedLet('u32', `${wordOf(slot.operand)} ^ ${mask}`)
            break
          case 'gate': {
            rules.lane += positions.length
            const zeros = slot.none.map(wordOf)
            text = sharedLet('u32', [...slot.all.map(wordOf), ...(zeros.length === 0 ? [] : [`(${zeros.length === 1 ? zeros[0]! : `(${zeros.join(' | ')})`} ^ ${mask})`])].join(' & '))
            break
          }
        }
        wordTexts.set(key, text)
        return text
      }
      const wordOfBits = (positions: readonly number[]) => wordOf(wordPlan(positions) ?? {kind: 'lanes', positions})

      // How a gate tests its values with a negative weight, which must all be 0. Gated reads with the same base go
      // into packed words. The bits, in lists of at most 32, are one word each where they are a word without building
      // it lane by lane. A comparison's inner gate and a whole number are tested on their own.
      function zeroTests(none: readonly number[]) {
        const groups = new Map<string, {gated: Gated, reads: IndexRow[]}>()
        const others: ({kind: 'comparison', inner: ReluRow} | {kind: 'number', position: number})[] = []
        const zeroBits: number[] = []
        for (const position of none) {
          const row = rowAt(position)
          switch (zeroTest(position)) {
            case 'read': {
              const read = row as IndexRow
              const gated = gatedRead(read)!
              const group = groups.get(gated.key)
              if (group === undefined) groups.set(gated.key, {gated, reads: [read]})
              else group.reads.push(read)
              break
            }
            case 'comparison':
              others.push({kind: 'comparison', inner: row as ReluRow})
              break
            case 'bit':
              zeroBits.push(position)
              break
            case 'number':
              others.push({kind: 'number', position})
              break
          }
        }
        const packs: {gated: Gated, reads: IndexRow[]}[] = []
        for (const {gated, reads} of groups.values()) {
          for (const list of wordLists(reads, read => read.position)) {
            // One gated read alone is an ordinary bit.
            if (list.length === 1) zeroBits.push(list[0]!.position)
            else packs.push({gated, reads: list})
          }
        }
        const bitLists = wordLists(zeroBits, position => position).map(bits => {
          const word = wordPlan(bits)
          return {bits, word: word !== undefined && isWhole(word) ? word : undefined}
        })
        return {others, packs, bitLists}
      }
      const testsWords = ({packs, bitLists}: ReturnType<typeof zeroTests>) => packs.length > 0 || bitLists.some(list => list.word !== undefined)

      // The Gate rule's statement: a u32 expression of 0 or 1.
      function gateText({all, none}: {all: number[], none: number[]}) {
        const factors = all.map(bitText)
        const {others, packs, bitLists} = zeroTests(none)
        for (const other of others) {
          if (other.kind === 'comparison') {
            // The Comparison rule: max(0, sum) is 0 exactly when sum <= 0.
            rules.comparison++
            factors.push(`u32(${atMostZero(other.inner.gateTerms, other.inner.gateBias)})`)
          } else {
            factors.push(`u32(${intText(other.position)} == 0i)`)
          }
        }
        for (const {gated, reads} of packs) factors.push(`u32((${packedWord(gated, reads)}) == 0u)`)
        // "None of these bits is 1": one comparison of a word with 0, or an OR of the bits.
        for (const {bits, word} of bitLists) {
          if (word !== undefined) factors.push(`u32(${wordOf(word)} == 0u)`)
          else factors.push(`(${bits.length === 1 ? bitText(bits[0]!) : `(${bits.map(bitText).join(' | ')})`} ^ 1u)`)
        }
        return factors.join(' & ')
      }

      // Today's f32 statement, from emit.ts's row writer.
      function writeFloat(row: Row): Held {
        const terms = row.kind === 'index' ? row.indexTerms : row.kind === 'reglu' ? [...row.gateTerms, ...row.terms] : row.terms
        for (const term of terms) floatText(term.position)
        writer.write(row)
        rules.float++
        return {text: `${prefix}v${row.position}`, type: 'float'}
      }

      function writeRow(row: Row): Held {
        const proof = proofs[row.position]
        const name = `${prefix}v${row.position}`
        if (!exact[row.position]!) return writeFloat(row)
        const bit = (expression: string): Held => {
          body.push(`let ${name}b: u32 = ${expression};`)
          return {text: `${name}b`, type: 'bit'}
        }
        // An i32 expression, held as a bit when the operation is proven to be one.
        const whole = (expression: string): Held => {
          if (isBit(proof!)) return bit(`u32(${expression})`)
          body.push(`let ${name}i: i32 = ${expression};`)
          return {text: `${name}i`, type: 'int'}
        }
        switch (row.kind) {
          case 'boolean': return writeFloat(row)
          case 'affine': {
            rules.sum++
            const negated = negatedBit(row)
            if (negated !== undefined) return bit(`${bitText(negated)} ^ 1u`)
            // A copy of a bit, e.g. a roundAffine of one term, stays a bit without a conversion.
            const copied = row.bias === 0 && row.terms.length === 1 && row.terms[0]!.weight === 1 && isBit(proofs[row.terms[0]!.position]!)
            return copied ? bit(bitText(row.terms[0]!.position)) : whole(intSum(row.terms, row.bias))
          }
          case 'reglu': {
            const gate = sumRange(proofs, row.gateTerms, row.gateBias)!
            if (row.terms.length > 0 || row.bias !== 1) {
              rules.product++
              const gateSum = intSum(row.gateTerms, row.gateBias)
              return whole(`${grouped(intSum(row.terms, row.bias))} * ${gate.minimum >= 0 ? grouped(gateSum) : `max(${gateSum}, 0i)`}`)
            }
            const form = conjunction(row)
            if (form !== undefined) {
              rules.gate++
              return bit(gateText(form))
            }
            rules.product++
            return isBit(proof!) ? bit(`u32(${intSum(row.gateTerms, row.gateBias)} > 0i)`) : whole(`max(${intSum(row.gateTerms, row.gateBias)}, 0i)`)
          }
          case 'index': {
            rules.read++
            const range = sumRange(proofs, row.indexTerms, row.indexBias)!
            const last = row.candidateCount - 1
            const index = sharedLet('i32', intSum(row.indexTerms, row.indexBias))
            let read = candidateText(row, range.minimum < 0 || range.maximum > last ? `clamp(${index}, 0i, ${last}i)` : index)
            if (row.aboveZero && range.maximum > last) read = `select(${read}, 0.0f, ${index} > ${last}i)`
            if (row.belowZero && range.minimum < 0) read = `select(${read}, 0.0f, ${index} < 0i)`
            if (proof === undefined) {
              body.push(`let ${name}: f32 = ${read};`)
              return {text: name, type: 'float'}
            }
            return isBit(proof) ? bit(`u32(${read})`) : whole(`i32(${read})`)
          }
        }
      }

      const outputPositions = positionsOfWires(program, stage.outputs.map(output => output.wire))
      const fieldWrites: string[] = []
      const windowWrites: string[] = []
      const documentWrites: string[] = []
      const registerWrites: string[] = []
      // The lanes of each field of bits and of each word of registers that this stage writes: the positions of proven
      // bits, or for a field that the loop program declares as bits without a proof, each lane's text.
      const lanesOfField = new Map<number, number[]>()
      const lanesOfWord = new Map<number, number[]>()
      const bitsOfField = new Map<number, string[]>()
      // A value as the type that a register or field is held in.
      const stored = (position: number, storage: Storage) => storage === 'bit' ? bitText(position) : storage === 'whole' ? intText(position) : floatText(position)
      stage.outputs.forEach(({write}, index) => {
        const position = outputPositions[index]!
        switch (write.kind) {
          case 'register': {
            const lane = laneOf.get(write.register)
            if (lane === undefined) {
              registerWrites.push(`r${write.register} = ${stored(position, registerStorage[write.register]!)};`)
            } else {
              const lanes = lanesOfWord.get(lane.word) ?? Array<number>(words[lane.word]!.length).fill(-1)
              lanes[lane.lane] = position
              lanesOfWord.set(lane.word, lanes)
            }
            break
          }
          case 'field': {
            const field = loop.fields[write.field]!
            const storage = fieldStorage[write.field]!
            if (fieldWords[write.field]! && storage === 'bit') {
              const lanes = lanesOfField.get(write.field) ?? Array<number>(field.width).fill(-1)
              lanes[write.component] = position
              lanesOfField.set(write.field, lanes)
            } else if (fieldWords[write.field]!) {
              const bits = bitsOfField.get(write.field) ?? Array<string>(field.width).fill('')
              bits[write.component] = `u32(${floatText(position)})`
              bitsOfField.set(write.field, bits)
            } else {
              fieldWrites.push(`field${write.field}[j * ${field.width}u + ${write.component}u] = ${stored(position, storage === 'bit' ? 'whole' : storage)};`)
            }
            break
          }
          case 'position':
            windowWrites.push(`instance_outputs_${write.group}[window_base_${write.group} + j] = ${floatText(position)};`)
            break
          case 'document':
            documentWrites.push(`instance_outputs_${write.group}[output_base_${write.group} + ${write.column}u] = ${floatText(position)};`)
            break
        }
      })
      for (const [field, bits] of bitsOfField) {
        invariant(bits.every(bit => bit !== ''), 'a stage that writes a field of bits writes every bit of it')
        fieldWrites.push(`field${field}[j] = ${inLanes(bits)};`)
      }
      for (const [field, lanes] of lanesOfField) {
        invariant(!lanes.includes(-1), 'a stage that writes a field of bits writes every bit of it')
        fieldWrites.push(`field${field}[j] = ${wordOfBits(lanes)};`)
      }
      // A word's registers have the same writers, so a stage that writes one of them writes all of them.
      for (const [word, lanes] of lanesOfWord) {
        invariant(!lanes.includes(-1), 'a stage that writes one register of a word writes all of them')
        registerWrites.push(`rw${word} = ${wordOfBits(lanes)};`)
      }
      const lines = [
        ...body, ...fieldWrites,
        ...(windowWrites.length > 0 ? ['if (in_window) {', ...windowWrites, '}'] : []),
        ...documentWrites,
        // Last, so that every statement above read the registers of the previous position.
        ...registerWrites,
      ]
      if (stage.repeat === 'once') stageLines.push(...lines)
      else if (stage.repeat === 'forward') stageLines.push('for (var j: u32 = 0u; j < positions; j++) {', ...lines, '}')
      else stageLines.push('for (var back: u32 = 0u; back < positions; back++) {', 'let j: u32 = positions - 1u - back;', ...lines, '}')
    })
    return {hoisted, packedCells, rules, rowCounts, stageLines, laneReads}
  }

  // The rest is loop-emit.ts's shader around the stages, with a type per register and field, and the packed words.
  const groups = loop.outputGroups
  if (loop.lengthCell !== undefined) {
    const fills = groups.flatMap((group, index) => group.kind === 'position' ? [`instance_outputs_${index}[window_base_${index} + j] = ${wgslFloat(group.fill)};`] : [])
    if (fills.length > 0) stageLines.push('if (in_window) {', `for (var j: u32 = positions; j < ${positions}u; j++) {`, ...fills, '}', '}')
  }
  for (const result of loop.results) {
    const lane = laneOfRegister(result.register)
    const register = lane === undefined ? `r${result.register}` : `((rw${lane.word} >> ${lane.lane}u) & 1u)`
    stageLines.push(`instance_outputs_${result.group}[output_base_${result.group} + ${result.column}u] = ${registerStorage[result.register] === 'float' ? register : `f32(${register})`};`)
  }
  let lengthName: string | undefined
  if (loop.lengthCell !== undefined) {
    const {buffer, column} = loop.lengthCell
    lengthName = `d${buffer}_${column}`
    hoisted.set(lengthName, `let ${lengthName}: f32 = ${bufferName(buffer)}[${baseName(buffer)} + ${column}u];`)
  }
  const packedBinding = 2 + loop.buffers.length + groups.length
  const main = [
    'let i: u32 = id.x;',
    'if (i >= params.count) {',
    'return;',
    '}',
    '_ = arrayLength(&shared_values);',
    ...loop.buffers.map((_, buffer) => `_ = arrayLength(&${bufferName(buffer)});`),
    'let row: u32 = params.first + i;',
    ...loop.buffers.map((buffer, index) => `let ${baseName(index)}: u32 = row * ${buffer.cellsPerDocument}u;`),
    ...groups.flatMap((group, index) => group.kind === 'document' ? [`let output_base_${index}: u32 = row * ${group.valuesPerDocument}u;`] : []),
    'let in_window: bool = row >= params.window_first && row - params.window_first < params.window_count;',
    ...groups.flatMap((group, index) => group.kind === 'position' ? [`let window_base_${index}: u32 = (row - params.window_first) * ${positions}u;`] : []),
    ...[...hoisted.values()],
    `let positions: u32 = ${lengthName === undefined ? `${positions}u` : `min(u32(${lengthName}), ${positions}u)`};`,
    ...loop.registers.flatMap((register, index) => {
      const storage = registerStorage[index]!
      if (laneOfRegister(index) !== undefined) return []
      return [storage === 'float' ? `var r${index}: f32 = ${wgslFloat(register.initial)};` : `var r${index}: ${storage === 'bit' ? `u32 = ${register.initial}u` : `i32 = ${wgslInt(register.initial)}`};`]
    }),
    ...registerWords.map((registers, word) => {
      let initial = 0
      registers.forEach((register, lane) => {
        if (loop.registers[register]!.initial === 1) initial |= 1 << lane
      })
      return `var rw${word}: u32 = ${initial >>> 0}u;`
    }),
    ...loop.fields.map((field, index) => fieldWords[index]!
      ? `var field${index}: array<u32, ${positions}>;`
      : `var field${index}: array<${fieldStorage[index] === 'float' ? 'f32' : 'i32'}, ${positions * field.width}>;`),
    ...stageLines,
  ]
  const code = [
    'struct Params {',
    '  count: u32,',
    '  window_first: u32,',
    '  window_count: u32,',
    '  first: u32,',
    '}',
    '@group(0) @binding(0) var<uniform> params: Params;',
    '@group(0) @binding(1) var<storage, read> shared_values: array<f32>;',
    ...loop.buffers.map((_, buffer) => `@group(0) @binding(${2 + buffer}) var<storage, read> ${bufferName(buffer)}: array<f32>;`),
    ...groups.map((_, group) => `@group(0) @binding(${2 + loop.buffers.length + group}) var<storage, read_write> instance_outputs_${group}: array<f32>;`),
    // After the bindings that loop-emit.ts's shader has, so that a runner binds those as before.
    ...(packedCells.length > 0 ? [`@group(0) @binding(${packedBinding}) var<storage, read> packed_words: array<u32>;`] : []),
    '',
    `@compute @workgroup_size(${workgroupSize})`,
    'fn main(@builtin(global_invocation_id) id: vec3<u32>) {',
    ...main,
    '}',
    '',
  ].join('\n')
  return {
    code, workgroupSize, statementCount: main.length, bytes: code.length, sharedLength: loop.sharedValues.length,
    cellsPerInstance: loop.buffers[0]!.cellsPerDocument, inputCells: loop.buffers.slice(1).map(buffer => buffer.cellsPerDocument),
    outputCounts: groups.map(group => group.valuesPerDocument), windowedGroups: groups.map(group => group.kind === 'position'), rowCounts,
    packedCells, rules,
  }
}
