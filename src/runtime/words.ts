// Words: exact u32 arithmetic next to the FP32 instructions. A Circuit computes over FP32, where whole numbers are exact
// up to 2^24 and every operation is affine + ReGLU; that's what makes a program differentiable, and what a counter, an
// id, a balance in cents or an index over 2^24 histories can't live with. A WordCircuit is the discrete half: every
// value is a u32, every operation is exact (wrapping where it says so), and nothing is differentiated. The two meet in a
// loop program (loop-program.ts), where a stage is either a Circuit or a WordCircuit, registers say which kind they
// hold, and a value crossing over is converted: f32(word), exact below 2^24, or u32(f32), the whole part of a value that
// isn't negative.
//
// Every operation's value is defined here by the reference evaluator, `evaluate`, and the two code writers, JavaScript
// (`compileJs`) and WGSL (`writeWgsl`), must give the same u32 for every input. They write the rules that WGSL leaves
// to the platform explicitly: division by zero gives 0 and the remainder by zero gives the dividend, so that
// a = b × (a / b) + a % b always holds, and a shift uses the low 5 bits of its amount. `words.test.ts` compares the
// three on random programs.
import {invariant} from './numeric-ui-core.ts'

declare const wordBrand: unique symbol
// A word's position: a source cell, then one per operation. Its own type, so a word can't be passed where an FP32
// wire is expected, or the other way around.
export type Word = number & {readonly [wordBrand]: true}

export type WordOp =
  | 'add' | 'sub' | 'mul' // wrapping, as u32 arithmetic is
  | 'monus' // a ∸ b: a - b when b <= a, else 0, the subtraction of natural numbers
  | 'div' | 'mod' // by zero: 0 and the dividend
  | 'min' | 'max'
  | 'and' | 'or' | 'xor' | 'shl' | 'shr' // shifts by the low 5 bits of b
  | 'eq' | 'ne' | 'lt' | 'le' // 1 or 0
  | 'not' // ~a
  | 'select' // c != 0 ? a : b, as args [c, a, b]
  | 'load' // the source cell at table + a, or 0 past the table's length

export type WordOperation =
  | {op: 'const', value: number}
  | {op: Exclude<WordOp, 'load'>, args: readonly Word[]}
  | {op: 'load', args: readonly [Word], table: number, length: number}

const arity: Record<Exclude<WordOp, 'load'>, number> = {
  add: 2, sub: 2, mul: 2, monus: 2, div: 2, mod: 2, min: 2, max: 2, and: 2, or: 2, xor: 2, shl: 2, shr: 2,
  eq: 2, ne: 2, lt: 2, le: 2, not: 1, select: 3,
}

// One operation's value, the definition every backend follows.
export function applyWord(op: Exclude<WordOp, 'load'>, a: number, b = 0, c = 0): number {
  switch (op) {
    case 'add': return (a + b) >>> 0
    case 'sub': return (a - b) >>> 0
    case 'mul': return Math.imul(a, b) >>> 0
    case 'monus': return a >= b ? a - b : 0
    case 'div': return b === 0 ? 0 : Math.floor(a / b)
    case 'mod': return b === 0 ? a : a % b
    case 'min': return Math.min(a, b)
    case 'max': return Math.max(a, b)
    case 'and': return (a & b) >>> 0
    case 'or': return (a | b) >>> 0
    case 'xor': return (a ^ b) >>> 0
    case 'shl': return (a << (b & 31)) >>> 0
    case 'shr': return a >>> (b & 31)
    case 'eq': return a === b ? 1 : 0
    case 'ne': return a !== b ? 1 : 0
    case 'lt': return a < b ? 1 : 0
    case 'le': return a <= b ? 1 : 0
    case 'not': return ~a >>> 0
    case 'select': return a !== 0 ? b : c
  }
}

const u32 = (value: number) => {
  invariant(Number.isInteger(value) && value >= 0 && value <= 0xffffffff, `${value} isn't a u32`)
  return value
}

export class WordCircuit {
  sourceSize: number
  operations: WordOperation[] = []
  // Identical operations share one word, e.g. two reads of the same constant.
  private keys = new Map<string, Word>()
  // The largest value declared for a source cell (assumeAtMost), and the largest each word can take, inferred from them.
  private declared = new Map<number, number>()
  private maxima: number[] | undefined
  private tableMaxima = new Map<string, number>()

  constructor(sourceSize: number) {
    invariant(Number.isInteger(sourceSize) && sourceSize >= 0, 'a word circuit has a whole number of source cells')
    this.sourceSize = sourceSize
  }

  get size() {
    return this.sourceSize + this.operations.length
  }

  source(index: number): Word {
    invariant(Number.isInteger(index) && index >= 0 && index < this.sourceSize, `source cell ${index} is outside the source table`)
    return index as Word
  }

  // A contract on a source cell, e.g. a loop's position is at most positions − 1: the code writers then read a table at
  // an index they can prove below its length without a bounds check. `evaluate` (the executor's path) checks it.
  assumeAtMost(word: Word, maximum: number) {
    invariant(word < this.sourceSize, 'a declared maximum is for a source cell')
    this.declared.set(word, Math.min(u32(maximum), this.declared.get(word) ?? 0xffffffff))
    this.maxima = undefined
    this.tableMaxima.clear()
  }

  // The largest value a word can take: declared for a source cell, inferred for an operation (0xffffffff when unknown).
  maximum(word: Word): number {
    if (this.maxima === undefined || this.maxima.length < this.size) {
      const top = 0xffffffff
      // Computed once, then extended for the operations appended since (a declaration starts it over).
      const m: number[] = this.maxima ?? Array.from({length: this.sourceSize}, (_, cell) => this.declared.get(cell) ?? top)
      const tables = this.tableMaxima
      const fill = (x: number) => (x >= 2 ** 31 ? top : 2 ** Math.ceil(Math.log2(x + 1)) - 1) // the smallest 2^k − 1 at or above x
      for (let index = m.length - this.sourceSize; index < this.operations.length; index++) {
        const operation = this.operations[index]!
        if (operation.op === 'const') { m.push(operation.value); continue }
        if (operation.op === 'load') {
          const key = `${operation.table},${operation.length}`
          let most = tables.get(key)
          if (most === undefined) { most = 0; for (let k = 0; k < operation.length; k++) most = Math.max(most, m[operation.table + k]!); tables.set(key, most) }
          m.push(most)
          continue
        }
        const [a, b, c] = operation.args.map(arg => m[arg]!) as [number, number, number]
        const constant = (arg: Word | undefined) => { const o = arg === undefined ? undefined : this.operations[arg - this.sourceSize]; return o?.op === 'const' ? o.value : undefined }
        switch (operation.op) {
          case 'add': m.push(a + b <= top ? a + b : top); break
          case 'mul': m.push(a * b <= top ? a * b : top); break
          case 'monus': { const d = constant(operation.args[1]); m.push(d !== undefined ? Math.max(0, a - d) : a); break }
          case 'div': { const d = constant(operation.args[1]); m.push(d !== undefined && d >= 1 ? Math.floor(a / d) : a); break }
          case 'shr': { const d = constant(operation.args[1]); m.push(d !== undefined ? a >>> (d & 31) : a); break }
          case 'mod': { const d = constant(operation.args[1]); m.push(d !== undefined && d >= 1 ? Math.min(a, d - 1) : a); break }
          case 'min': case 'and': m.push(Math.min(a, b)); break
          case 'max': m.push(Math.max(a, b)); break
          case 'or': case 'xor': m.push(fill(Math.max(a, b))); break
          case 'eq': case 'ne': case 'lt': case 'le': m.push(1); break
          case 'select': m.push(Math.max(b, c)); break
          default: m.push(top) // sub, shl, not
        }
      }
      this.maxima = m
    }
    return this.maxima[word]!
  }

  private append(operation: WordOperation): Word {
    if (operation.op !== 'const') for (const arg of operation.args) invariant(Number.isInteger(arg) && arg >= 0 && arg < this.size, `word ${arg} doesn't exist yet`)
    const key = operation.op === 'const' ? `c${operation.value}` : operation.op === 'load' ? `l${operation.table},${operation.length},${operation.args[0]}` : `${operation.op}${operation.args.join(',')}`
    const existing = this.keys.get(key)
    if (existing !== undefined) return existing
    const word = this.size as Word
    this.operations.push(operation)
    this.keys.set(key, word)
    return word
  }

  constant(value: number): Word {
    return this.append({op: 'const', value: u32(value)})
  }

  // Any operation but 'load' and 'const', e.g. circuit.op('monus', budget, spent). Operations on constants fold, and
  // identities that hold for every value (x + 0, x × 1, x & 1 for a word that is at most 1, select on a constant, ...)
  // return the operand, so builder code can be written plainly and still give short shaders.
  op(op: Exclude<WordOp, 'load'>, ...args: Word[]): Word {
    invariant(args.length === arity[op], `${op} takes ${arity[op]} words`)
    for (const arg of args) invariant(Number.isInteger(arg) && arg >= 0 && arg < this.size, `word ${arg} doesn't exist yet`)
    const value = (w: Word | undefined) => { const o = w === undefined ? undefined : this.operations[w - this.sourceSize]; return o?.op === 'const' ? o.value : undefined }
    const [a, b, c] = args as [Word, Word, Word]
    const [va, vb, vc] = [value(a), value(b), value(c)]
    if (args.every(arg => value(arg) !== undefined)) return this.constant(applyWord(op, va!, vb ?? 0, vc ?? 0))
    const top = 0xffffffff
    switch (op) {
      case 'add': if (va === 0) return b; if (vb === 0) return a; break
      case 'sub': case 'shl': case 'shr': case 'xor': case 'or': case 'monus': if (vb === 0) return a; if ((op === 'or' || op === 'xor') && va === 0) return b; if (op === 'monus' && va === 0) return a; break
      case 'mul': if (va === 0 || vb === 0) return this.constant(0); if (va === 1) return b; if (vb === 1) return a; break
      case 'div': if (vb === 1) return a; break
      case 'mod': if (vb === 1) return this.constant(0); break
      case 'and': if (va === 0 || vb === 0) return this.constant(0); if (va === top || (va !== undefined && this.maximum(b) <= va && (va & (va + 1)) === 0)) return b; if (vb === top || (vb !== undefined && this.maximum(a) <= vb && (vb & (vb + 1)) === 0)) return a; break
      case 'min': if (va === 0 || vb === 0) return this.constant(0); if (va === top) return b; if (vb === top) return a; if (a === b) return a; break
      case 'max': if (va === 0) return b; if (vb === 0) return a; if (a === b) return a; break
      case 'select': if (va !== undefined) return va !== 0 ? b : c; if (b === c) return b; if (vb === 1 && vc === 0 && this.maximum(a) <= 1) return a; break
      case 'eq': case 'le': if (a === b) return this.constant(1); if (op === 'le' && (va === 0 || vb === top || (vb !== undefined && this.maximum(a) <= vb))) return this.constant(1); if (op === 'eq' && ((vb !== undefined && vb > this.maximum(a)) || (va !== undefined && va > this.maximum(b)))) return this.constant(0); break
      case 'not': break // no identity: ~a is a new word
      case 'ne': case 'lt': if (a === b) return this.constant(0); if (op === 'lt' && (vb === 0 || va === top)) return this.constant(0); if (op === 'lt' && vb !== undefined && this.maximum(a) < vb) return this.constant(1); if (op === 'ne' && ((vb !== undefined && vb > this.maximum(a)) || (va !== undefined && va > this.maximum(b)))) return this.constant(1); break
    }
    return this.append({op, args})
  }

  add(a: Word, b: Word) { return this.op('add', a, b) }
  sub(a: Word, b: Word) { return this.op('sub', a, b) }
  mul(a: Word, b: Word) { return this.op('mul', a, b) }
  monus(a: Word, b: Word) { return this.op('monus', a, b) }
  div(a: Word, b: Word) { return this.op('div', a, b) }
  mod(a: Word, b: Word) { return this.op('mod', a, b) }
  min(a: Word, b: Word) { return this.op('min', a, b) }
  max(a: Word, b: Word) { return this.op('max', a, b) }
  and(a: Word, b: Word) { return this.op('and', a, b) }
  or(a: Word, b: Word) { return this.op('or', a, b) }
  xor(a: Word, b: Word) { return this.op('xor', a, b) }
  shl(a: Word, b: Word) { return this.op('shl', a, b) }
  shr(a: Word, b: Word) { return this.op('shr', a, b) }
  eq(a: Word, b: Word) { return this.op('eq', a, b) }
  ne(a: Word, b: Word) { return this.op('ne', a, b) }
  lt(a: Word, b: Word) { return this.op('lt', a, b) }
  le(a: Word, b: Word) { return this.op('le', a, b) }
  not(a: Word) { return this.op('not', a) }
  select(condition: Word, whenTrue: Word, whenFalse: Word) { return this.op('select', condition, whenTrue, whenFalse) }

  // The source cell at table + index, or 0 when index >= length, e.g. a coefficient of candidate j in a shared table.
  load(table: number, length: number, index: Word): Word {
    invariant(Number.isInteger(table) && Number.isInteger(length) && table >= 0 && length >= 1 && table + length <= this.sourceSize, 'a table lies within the source cells')
    return this.append({op: 'load', args: [index], table, length})
  }

  // Every word's value, from the source cells: the definition the code writers follow.
  evaluate(sources: Uint32Array): Uint32Array {
    invariant(sources.length === this.sourceSize, 'the sources have the circuit\'s length')
    for (const [cell, maximum] of this.declared) invariant(sources[cell]! <= maximum, `source cell ${cell} holds ${sources[cell]}, above its declared maximum ${maximum}`)
    const values = new Uint32Array(this.size)
    values.set(sources)
    this.operations.forEach((operation, index) => {
      const at = this.sourceSize + index
      if (operation.op === 'const') values[at] = operation.value
      else if (operation.op === 'load') { const i = values[operation.args[0]]!; values[at] = i < operation.length ? values[operation.table + i]! : 0 }
      else values[at] = applyWord(operation.op, values[operation.args[0]!]!, values[operation.args[1] ?? 0], values[operation.args[2] ?? 0])
    })
    return values
  }

  // The words `roots` need, in order: an operation nothing needed is skipped by the writers.
  live(roots: readonly Word[]): boolean[] {
    const live = Array<boolean>(this.size).fill(false)
    const tables = new Set<string>() // each distinct table's cells are marked once, however many loads read it
    const stack = [...roots]
    while (stack.length > 0) {
      const word = stack.pop()!
      if (live[word]) continue
      live[word] = true
      const operation = this.operations[word - this.sourceSize]
      if (operation !== undefined && operation.op !== 'const') stack.push(...operation.args)
      if (operation?.op === 'load' && !tables.has(`${operation.table}:${operation.length}`)) {
        tables.add(`${operation.table}:${operation.length}`)
        for (let k = 0; k < operation.length; k++) live[operation.table + k] = true // table cells are source cells
      }
    }
    return live
  }

  // JavaScript statements that compute `roots` into `values` (a Uint32Array of the circuit's size whose source cells the
  // caller filled), several times faster than `evaluate`.
  compileJs(roots: readonly Word[]): (values: Uint32Array) => void {
    const live = this.live(roots)
    // A constant is written as its literal where it is read; roots still get their values.
    const rootSet = new Set<number>(roots)
    const name = (word: number) => { const o = this.operations[word - this.sourceSize]; return o?.op === 'const' ? String(o.value) : `v[${word}]` }
    const lines: string[] = []
    this.operations.forEach((operation, index) => {
      const at = this.sourceSize + index
      if (!live[at] || (operation.op === 'const' && !rootSet.has(at))) return
      lines.push(`v[${at}] = ${jsExpression(operation, name, this.inRange(operation))};`)
    })
    // The statements come from the circuit's own operations and word indexes, never from data, so compiling them is safe.
    // oxlint-disable-next-line typescript/no-implied-eval
    return new Function('v', lines.join('\n')) as (values: Uint32Array) => void
  }

  // WGSL `let` statements for `roots`, one per live operation, given how a source cell is read. Returns the statements and
  // the text of each word.
  writeWgsl(roots: readonly Word[], options: {prefix: string, source: (index: number) => string}) {
    const live = this.live(roots)
    const body: string[] = []
    const names = new Map<number, string>()
    const name = (word: number): string => {
      if (word < this.sourceSize) {
        let text = names.get(word)
        if (text === undefined) { text = options.source(word); names.set(word, text) }
        return text
      }
      const o = this.operations[word - this.sourceSize]
      return o?.op === 'const' ? `${o.value}u` : `${options.prefix}w${word}` // constants are literals where read
    }
    this.operations.forEach((operation, index) => {
      const at = this.sourceSize + index
      if (!live[at] || operation.op === 'const') return
      body.push(`let ${options.prefix}w${at}: u32 = ${wgslExpression(operation, name, options.source, this.inRange(operation))};`)
    })
    return {body, name}
  }

  // A load whose index is provably below its table's length needs no bounds check.
  private inRange(operation: WordOperation) {
    return operation.op === 'load' && this.maximum(operation.args[0]) < operation.length
  }
}

const jsOps: Partial<Record<WordOp, (a: string, b: string, c: string) => string>> = {
  add: (a, b) => `(${a} + ${b}) >>> 0`, sub: (a, b) => `(${a} - ${b}) >>> 0`, mul: (a, b) => `Math.imul(${a}, ${b}) >>> 0`,
  monus: (a, b) => `(${a} >= ${b} ? ${a} - ${b} : 0)`, div: (a, b) => `(${b} === 0 ? 0 : Math.floor(${a} / ${b}))`, mod: (a, b) => `(${b} === 0 ? ${a} : ${a} % ${b})`,
  min: (a, b) => `Math.min(${a}, ${b})`, max: (a, b) => `Math.max(${a}, ${b})`,
  and: (a, b) => `(${a} & ${b}) >>> 0`, or: (a, b) => `(${a} | ${b}) >>> 0`, xor: (a, b) => `(${a} ^ ${b}) >>> 0`,
  shl: (a, b) => `(${a} << (${b} & 31)) >>> 0`, shr: (a, b) => `${a} >>> (${b} & 31)`,
  eq: (a, b) => `(${a} === ${b} ? 1 : 0)`, ne: (a, b) => `(${a} !== ${b} ? 1 : 0)`, lt: (a, b) => `(${a} < ${b} ? 1 : 0)`, le: (a, b) => `(${a} <= ${b} ? 1 : 0)`,
  not: a => `~${a} >>> 0`, select: (c, a, b) => `(${c} !== 0 ? ${a} : ${b})`,
}
function jsExpression(operation: WordOperation, name: (word: number) => string, inRange: boolean): string {
  if (operation.op === 'const') return String(operation.value)
  if (operation.op === 'load') { const i = name(operation.args[0]); return inRange ? `v[${operation.table} + ${i}]` : `(${i} < ${operation.length} ? v[${operation.table} + ${i}] : 0)` }
  const [a, b, c] = operation.args.map(name)
  return jsOps[operation.op]!(a!, b ?? '', c ?? '')
}

const wgslOps: Partial<Record<WordOp, (a: string, b: string, c: string) => string>> = {
  add: (a, b) => `${a} + ${b}`, sub: (a, b) => `${a} - ${b}`, mul: (a, b) => `${a} * ${b}`,
  monus: (a, b) => `select(0u, ${a} - ${b}, ${a} >= ${b})`,
  // WGSL defines both at zero too, but differently across versions of the spec: written out, they are this module's rule.
  div: (a, b) => `select(${a} / max(${b}, 1u), 0u, ${b} == 0u)`, mod: (a, b) => `select(${a} % max(${b}, 1u), ${a}, ${b} == 0u)`,
  min: (a, b) => `min(${a}, ${b})`, max: (a, b) => `max(${a}, ${b})`,
  and: (a, b) => `${a} & ${b}`, or: (a, b) => `${a} | ${b}`, xor: (a, b) => `${a} ^ ${b}`,
  shl: (a, b) => `${a} << (${b} & 31u)`, shr: (a, b) => `${a} >> (${b} & 31u)`,
  eq: (a, b) => `u32(${a} == ${b})`, ne: (a, b) => `u32(${a} != ${b})`, lt: (a, b) => `u32(${a} < ${b})`, le: (a, b) => `u32(${a} <= ${b})`,
  not: a => `~${a}`, select: (c, a, b) => `select(${b}, ${a}, ${c} != 0u)`,
}
function wgslExpression(operation: WordOperation, name: (word: number) => string, source: (index: number) => string, inRange: boolean): string {
  if (operation.op === 'const') return `${operation.value}u`
  if (operation.op === 'load') {
    // A table of source cells is read through the caller's `source`, given an index expression: the cells must sit at
    // consecutive positions in what `source` reads, e.g. shared words.
    const i = name(operation.args[0])
    const first = source(operation.table)
    const match = /^(.*)\[(\d+)u\]$/.exec(first)
    invariant(match !== null, 'a loaded table is read from an array the shader indexes, e.g. shared_words[17u]')
    return inRange ? `${match[1]}[${match[2]}u + ${i}]` : `select(0u, ${match[1]}[${match[2]}u + min(${i}, ${operation.length - 1}u)], ${i} < ${operation.length}u)`
  }
  const [a, b, c] = operation.args.map(arg => `(${name(arg)})`)
  // A divisor that is a nonzero literal needs no rule at zero.
  if ((operation.op === 'div' || operation.op === 'mod') && /^\(\d+u\)$/.test(b!) && b !== '(0u)') return `${a} ${operation.op === 'div' ? '/' : '%'} ${b}`
  return wgslOps[operation.op]!(a!, b ?? '', c ?? '')
}
