// One WGSL compute shader from a loop program (loop-program.ts): one invocation runs one document, and each looped
// stage's operations are written once, inside a `for` loop over the document's positions. So the shader's length is
// the work of one position, whatever the number of positions, e.g. 496 statements for regex search's step at 8 states.
//
// What the shader reads and writes:
//   shared_values       every stage's source table, e.g. a pattern's cells, uploaded per search
//   instance_sources    buffer 0 of the loop program, e.g. 513 cells per document
//   instance_inputs_b   buffer b, e.g. the display kind of every code point, which another shader wrote
//   instance_outputs_g  output group g. A 'document' group has a row per document. A 'position' group has a row only
//                       for the documents params.window_first to window_first + window_count - 1, e.g. the ones on screen
//   params              count documents run, from document params.first on, e.g. every document, or only the window
//
// A register is a `var` of the function, and a looped stage reads it at the start of its body and assigns it at the
// end. A field is an array of the function: f32 values, or for a field of bits one u32 word per position, which keeps
// e.g. 32 bits × 512 positions in 2 KB instead of 64 KB. The stages' statements are emit.ts's, one `let` per operation,
// without guards: a loop program here holds only integers, see emit.ts's note on exactness.
//
// Words (runtime/words.ts): a word stage's statements are its WordCircuit's, `let a_w12: u32 = ...`. A program with any
// words declares shared_values as array<u32>, since words keep their bits there, and an FP32 stage reads a shared cell
// as bitcast<f32>(...), which gives back the same f32. A u32 register is a `var r3: u32`, a u32 output group is an
// array<u32>, and a value crossing kinds is converted: f32(word), u32(value).
import {type Wire, invariant} from '../runtime/numeric-ui-core.ts'
import {type Row, createRowWriter, flattenProgram, positionsOfWires, wgslFloat, wgslInt} from './emit.ts'
import {type AnyStage, type LoopProgram, type WordBinding, isWordStage} from './loop-program.ts'

export type LoopWgslModule = {
  code: string
  workgroupSize: number
  // The lines of main, with each looped stage's lines counted once.
  statementCount: number
  bytes: number
  sharedLength: number
  // The cells per document of buffer 0, and of each later buffer.
  cellsPerInstance: number
  inputCells: number[]
  // Values per document, or per document of the window, in each output buffer.
  outputCounts: number[]
  windowedGroups: boolean[]
  rowCounts: {affine: number, reglu: number, boolean: number, index: number}
}

// The rows in an order where each row comes right after the rows it reads, found by walking back from the outputs:
// e.g. the 32 table reads of one state's sum sit next to that sum. The compiled program lists a whole layer of reads
// before the layer of sums, which leaves e.g. 1,024 values alive at once at 32 states, more than a GPU thread has
// registers. Every row computes the same value in either order.
function orderRowsByUse(rows: readonly Row[], sourceSize: number, outputPositions: readonly number[]): Row[] {
  const rowAt = new Map(rows.map(row => [row.position, row]))
  const ordered: Row[] = []
  const visited = new Set<number>()
  const visit = (position: number) => {
    if (position < sourceSize || visited.has(position)) return
    visited.add(position)
    const row = rowAt.get(position)!
    const terms = row.kind === 'index' ? row.indexTerms : row.kind === 'reglu' ? [...row.gateTerms, ...row.terms] : row.terms
    for (const term of terms) visit(term.position)
    ordered.push(row)
  }
  for (const position of outputPositions) visit(position)
  invariant(ordered.length === rows.length, 'every row of a stage leads to one of its outputs')
  return ordered
}

type LoopWgslOptions = {
  workgroupSize?: number
  // 'use', the default, writes each row right after the rows it reads. 'layers' keeps the compiled program's order.
  order?: 'use' | 'layers'
}

export function emitLoopWgsl(loop: LoopProgram<AnyStage>, options: LoopWgslOptions = {}): LoopWgslModule {
  const workgroupSize = options.workgroupSize ?? 64
  const {positions} = loop
  const bufferName = (buffer: number) => buffer === 0 ? 'instance_sources' : `instance_inputs_${buffer}`
  const baseName = (buffer: number) => buffer === 0 ? 'source_base' : `input_base_${buffer}`
  const hoistedShared = new Set<number>()
  const hoistedCells = new Map<string, string>()
  const cellName = (buffer: number, column: number) => {
    const name = `d${buffer}_${column}`
    hoistedCells.set(name, `let ${name}: f32 = ${bufferName(buffer)}[${baseName(buffer)} + ${column}u];`)
    return name
  }
  const rowCounts = {affine: 0, reglu: 0, boolean: 0, index: 0}
  const stageLines: string[] = []
  // A loop over a fixed count says the count in its header: Metal's compiler treats a `let` bound as unknown, which
  // cost a word-stage enumeration of 2^31 cases a third of its time on an M2 (39.5 ms against 27.4 ms).
  const bound = loop.lengthCell === undefined ? `${positions}u` : 'positions'
  const wordRegister = loop.registers.map(register => register.type === 'u32')
  const wordGroup = loop.outputGroups.map(group => group.type === 'u32')
  const hasWords = loop.stages.some(isWordStage) || wordRegister.some(Boolean) || wordGroup.some(Boolean)
  const sharedRead = (position: number) => (hasWords ? `bitcast<f32>(shared_values[${position}u])` : `shared_values[${position}u]`)
  // A value written where the other kind is held: f32(word) or u32(value).
  const asStored = (value: string, fromWord: boolean, toWord: boolean) => (fromWord === toWord ? value : toWord ? `u32(${value})` : `f32(${value})`)

  loop.stages.forEach((stage, stageIndex) => {
    const prefix = `${String.fromCharCode(97 + stageIndex)}_`
    if (isWordStage(stage)) {
      // Each source cell is read once, converted to a u32 where it holds an f32.
      const perPosition: string[] = []
      const wordsRead = new Set<number>()
      const read = (position: number, binding: WordBinding | undefined): string => {
        if (binding === undefined) return `shared_values[${stage.sharedOffset + position}u]`
        switch (binding.kind) {
          case 'register': return wordRegister[binding.register] ? `r${binding.register}` : `u32(r${binding.register})`
          case 'cell': return `u32(${cellName(binding.buffer, binding.column)})`
          case 'document': return 'row'
          case 'position': return 'j'
          case 'item': {
            const name = `${prefix}c${position}`
            perPosition.push(`let ${name}: u32 = u32(${bufferName(binding.buffer)}[${baseName(binding.buffer)} + j]);`)
            return name
          }
          case 'field': {
            const field = loop.fields[binding.field]!
            if (!field.bits) return `u32(field${binding.field}[j * ${field.width}u + ${binding.component}u])`
            const word = `${prefix}fw${binding.field}`
            if (!wordsRead.has(binding.field)) { wordsRead.add(binding.field); perPosition.push(`let ${word}: u32 = field${binding.field}[j];`) }
            return `((${word} >> ${binding.component}u) & 1u)`
          }
        }
      }
      const written = stage.circuit.writeWgsl(stage.outputs.map(output => output.wire), {prefix, source: position => read(position, stage.bindings.get(position as never))})
      const body = [...perPosition, ...written.body]
      const registerWrites: string[] = [], late: string[] = []
      let stop: string | undefined
      const bitsOfField = new Map<number, string[]>()
      for (const {wire, write} of stage.outputs) {
        const value = written.name(wire)
        switch (write.kind) {
          case 'stop': stop = value; break
          case 'register': registerWrites.push(`r${write.register} = ${asStored(value, true, wordRegister[write.register]!)};`); break
          case 'field': {
            const field = loop.fields[write.field]!
            if (field.bits) {
              const bits = bitsOfField.get(write.field) ?? Array<string>(field.width).fill('')
              bits[write.component] = write.component === 0 ? `(${value})` : `((${value}) << ${write.component}u)`
              bitsOfField.set(write.field, bits)
            } else late.push(`field${write.field}[j * ${field.width}u + ${write.component}u] = f32(${value});`)
            break
          }
          case 'position': late.push(`if (in_window) { instance_outputs_${write.group}[window_base_${write.group} + j] = ${asStored(value, true, wordGroup[write.group]!)}; }`); break
          case 'document': late.push(`instance_outputs_${write.group}[output_base_${write.group} + ${write.column}u] = ${asStored(value, true, wordGroup[write.group]!)};`); break
        }
      }
      for (const [field, bits] of bitsOfField) late.push(`field${field}[j] = ${bits.join(' | ')};`)
      // The stop word is read before the registers change, then the loop ends after this position's writes.
      const all = stop === undefined ? [...body, ...late, ...registerWrites] : [...body, `let ${prefix}stop: bool = ${stop} != 0u;`, ...late, ...registerWrites, `if (${prefix}stop) { break; }`]
      if (stage.repeat === 'once') stageLines.push(...all)
      else if (stage.repeat === 'forward') stageLines.push(`for (var j: u32 = 0u; j < ${bound}; j++) {`, ...all, '}')
      else stageLines.push(`for (var back: u32 = 0u; back < ${bound}; back++) {`, 'let j: u32 = positions - 1u - back;', ...all, '}')
      return
    }
    const {program} = stage
    // The `let`s at the start of a looped stage's body: the items and fields of this position.
    const perPosition: string[] = []
    const names = new Map<number, string>()
    const wordsRead = new Set<number>()
    const writer = createRowWriter(program.sourceSize, {
      prefix,
      source: position => {
        const existing = names.get(position)
        if (existing !== undefined) return existing
        const binding = stage.bindings.get(position as Wire)
        let name: string
        if (binding === undefined) {
          hoistedShared.add(stage.sharedOffset + position)
          name = `s${stage.sharedOffset + position}`
        } else if (binding.kind === 'register') {
          name = wordRegister[binding.register] ? `f32(r${binding.register})` : `r${binding.register}`
        } else if (binding.kind === 'cell') {
          name = cellName(binding.buffer, binding.column)
        } else if (binding.kind === 'item') {
          name = `${prefix}c${position}`
          perPosition.push(`let ${name}: f32 = ${bufferName(binding.buffer)}[${baseName(binding.buffer)} + j];`)
        } else {
          const field = loop.fields[binding.field]!
          name = `${prefix}f${position}`
          if (field.bits) {
            const word = `${prefix}w${binding.field}`
            if (!wordsRead.has(binding.field)) {
              wordsRead.add(binding.field)
              perPosition.push(`let ${word}: u32 = field${binding.field}[j];`)
            }
            perPosition.push(`let ${name}: f32 = f32((${word} >> ${binding.component}u) & 1u);`)
          } else {
            perPosition.push(`let ${name}: f32 = field${binding.field}[j * ${field.width}u + ${binding.component}u];`)
          }
        }
        names.set(position, name)
        return name
      },
      candidate: (row, clamped) => {
        for (let candidate = 0; candidate < row.candidateCount; candidate++) {
          invariant(!stage.bindings.has((row.candidateBase + candidate * row.candidateStride) as Wire), 'in a loop program, an index read\'s candidates are shared cells')
        }
        const at = `u32(${stage.sharedOffset + row.candidateBase}i + ${clamped} * ${wgslInt(row.candidateStride)})`
        return hasWords ? `bitcast<f32>(shared_values[${at}])` : `shared_values[${at}]`
      },
    })
    const outputPositions = positionsOfWires(program, stage.outputs.map(output => output.wire))
    const rows = flattenProgram(program)
    for (const row of options.order === 'layers' ? rows : orderRowsByUse(rows, program.sourceSize, outputPositions)) writer.write(row)
    for (const kind of ['affine', 'reglu', 'boolean', 'index'] as const) rowCounts[kind] += writer.rowCounts[kind]

    const fieldWrites: string[] = []
    const windowWrites: string[] = []
    const documentWrites: string[] = []
    const registerWrites: string[] = []
    const bitsOfField = new Map<number, string[]>()
    stage.outputs.forEach(({write}, index) => {
      const value = writer.name(outputPositions[index]!)
      switch (write.kind) {
        case 'register':
          registerWrites.push(`r${write.register} = ${asStored(value, false, wordRegister[write.register]!)};`)
          break
        case 'field': {
          const field = loop.fields[write.field]!
          if (field.bits) {
            const bits = bitsOfField.get(write.field) ?? Array<string>(field.width).fill('')
            bits[write.component] = write.component === 0 ? `u32(${value})` : `(u32(${value}) << ${write.component}u)`
            bitsOfField.set(write.field, bits)
          } else {
            fieldWrites.push(`field${write.field}[j * ${field.width}u + ${write.component}u] = ${value};`)
          }
          break
        }
        case 'position':
          windowWrites.push(`instance_outputs_${write.group}[window_base_${write.group} + j] = ${asStored(value, false, wordGroup[write.group]!)};`)
          break
        case 'document':
          documentWrites.push(`instance_outputs_${write.group}[output_base_${write.group} + ${write.column}u] = ${asStored(value, false, wordGroup[write.group]!)};`)
          break
      }
    })
    for (const [field, bits] of bitsOfField) {
      invariant(bits.every(bit => bit !== ''), 'a stage that writes a field of bits writes every bit of it')
      fieldWrites.push(`field${field}[j] = ${bits.join(' | ')};`)
    }
    const body = [
      ...perPosition, ...writer.body, ...fieldWrites,
      ...(windowWrites.length > 0 ? ['if (in_window) {', ...windowWrites, '}'] : []),
      ...documentWrites,
      // Last, so that every statement above read the registers of the previous position.
      ...registerWrites,
    ]
    if (stage.repeat === 'once') {
      invariant(perPosition.length === 0, 'a stage that runs once reads no item and no field')
      stageLines.push(...body)
    } else if (stage.repeat === 'forward') {
      stageLines.push(`for (var j: u32 = 0u; j < ${bound}; j++) {`, ...body, '}')
    } else {
      stageLines.push(`for (var back: u32 = 0u; back < ${bound}; back++) {`, 'let j: u32 = positions - 1u - back;', ...body, '}')
    }
  })

  const groups = loop.outputGroups
  if (loop.lengthCell !== undefined) {
    // The positions past the text's end, which the loops skipped.
    const fills = groups.flatMap((group, index) => group.kind === 'position' ? [`instance_outputs_${index}[window_base_${index} + j] = ${wordGroup[index] ? `${group.fill >>> 0}u` : wgslFloat(group.fill)};`] : [])
    if (fills.length > 0) stageLines.push('if (in_window) {', `for (var j: u32 = positions; j < ${positions}u; j++) {`, ...fills, '}', '}')
  }
  for (const result of loop.results) stageLines.push(`instance_outputs_${result.group}[output_base_${result.group} + ${result.column}u] = ${asStored(`r${result.register}`, wordRegister[result.register]!, wordGroup[result.group]!)};`)

  const lengthName = loop.lengthCell === undefined ? undefined : cellName(loop.lengthCell.buffer, loop.lengthCell.column)
  const main = [
    'let i: u32 = id.x;',
    'if (i >= params.count) {',
    'return;',
    '}',
    // Static uses keep every binding in the automatic pipeline layout.
    '_ = arrayLength(&shared_values);',
    ...loop.buffers.map((_, buffer) => `_ = arrayLength(&${bufferName(buffer)});`),
    // Invocation i runs document params.first + i, e.g. only the documents on screen after a scroll.
    'let row: u32 = params.first + i;',
    ...loop.buffers.map((buffer, index) => `let ${baseName(index)}: u32 = row * ${buffer.cellsPerDocument}u;`),
    ...groups.flatMap((group, index) => group.kind === 'document' ? [`let output_base_${index}: u32 = row * ${group.valuesPerDocument}u;`] : []),
    'let in_window: bool = row >= params.window_first && row - params.window_first < params.window_count;',
    // Only read where in_window holds, so the subtraction's wraparound for an earlier document doesn't matter.
    ...groups.flatMap((group, index) => group.kind === 'position' ? [`let window_base_${index}: u32 = (row - params.window_first) * ${positions}u;`] : []),
    ...[...hoistedShared].sort((left, right) => left - right).map(position => `let s${position}: f32 = ${sharedRead(position)};`),
    ...[...hoistedCells.values()],
    // The text's length is a whole number of at most 1,048,576, so the conversion is exact.
    `let positions: u32 = ${lengthName === undefined ? `${positions}u` : `min(u32(${lengthName}), ${positions}u)`};`,
    ...loop.registers.map((register, index) => (wordRegister[index] ? `var r${index}: u32 = ${register.initial >>> 0}u;` : `var r${index}: f32 = ${wgslFloat(register.initial)};`)),
    ...loop.fields.map((field, index) => field.bits
      ? `var field${index}: array<u32, ${positions}>;`
      : `var field${index}: array<f32, ${positions * field.width}>;`),
    ...stageLines,
  ]
  for (const field of loop.fields) invariant(!field.bits || field.width <= 32, 'a field of bits holds at most 32 bits per position')
  const code = [
    'struct Params {',
    '  count: u32,',
    '  window_first: u32,',
    '  window_count: u32,',
    '  first: u32,',
    '}',
    '@group(0) @binding(0) var<uniform> params: Params;',
    `@group(0) @binding(1) var<storage, read> shared_values: array<${hasWords ? 'u32' : 'f32'}>;`,
    ...loop.buffers.map((_, buffer) => `@group(0) @binding(${2 + buffer}) var<storage, read> ${bufferName(buffer)}: array<f32>;`),
    ...groups.map((_, group) => `@group(0) @binding(${2 + loop.buffers.length + group}) var<storage, read_write> instance_outputs_${group}: array<${wordGroup[group] ? 'u32' : 'f32'}>;`),
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
  }
}
