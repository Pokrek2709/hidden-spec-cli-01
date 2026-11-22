// main.ts is mend, the CLI SPEC.md specifies. It applies AI-written edits in four
// formats (unified diffs with wrong line numbers, SEARCH/REPLACE blocks, V4A patches
// and JSON tool calls) to a tree of text files, finding the lines an edit names at
// the first of three matching tiers, and changing every file or none. Node 22, the
// standard library alone; run it as `node main.ts [options] [PATCH]`.
//
// One parser per format, one shared line applier: every format parses to edits over
// file lines. Unified hunks take the occurrence nearest their (wrong) line number,
// V4A chunks walk a cursor from their anchor, SEARCH/REPLACE blocks must match
// exactly once, JSON edits are exact. Every edit runs against an in-memory tree
// first; the tree is only written when no edit failed, and a failed edit changes
// nothing while the edits after it still run.
//
// Keep the hand-over at the end and runner.ts as they are: in a trial the kit's
// runner reads the checks from the env and runs this file on each one.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

const FORMATS = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

class Exit2 {
  readonly line: string;
  constructor(line: string) {
    this.line = line;
  }
}
class Fail {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const blank = (s: string): boolean => s.trim() === '';

function patchLines(text: string): string[] {
  if (text === '') return [];
  const lines = text.split('\n');
  if (text.endsWith('\n')) lines.pop();
  return lines.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}

function safe(p: string): boolean {
  return p !== '' && !p.startsWith('/') && !p.includes('\\') && p.split('/').every((s) => s !== '' && s !== '.' && s !== '..');
}

/** The tree as the edits leave it: changed paths in `now` (null: deleted), the rest read from disk. */
class Tree {
  private readonly orig = new Map<string, string | null>();
  readonly now = new Map<string, string | null>();
  /** Successful move edits, as [old, new] pairs, for the `R` output lines. */
  readonly moves: Array<[string, string]> = [];
  original(p: string): string | null {
    if (!this.orig.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.orig.set(p, v);
    }
    return this.orig.get(p)!;
  }
  get(p: string): string | null {
    return this.now.has(p) ? this.now.get(p)! : this.original(p);
  }
}

type Edit = (tree: Tree) => void;

/** A file as lines: its line break, its lines, and whether it ends with a break. */
interface FileModel {
  br: string;
  lines: string[];
  final: boolean;
}

function toModel(text: string): FileModel {
  if (text === '') return { br: '\n', lines: [], final: false };
  const nl = text.indexOf('\n');
  const br = nl > 0 && text[nl - 1] === '\r' ? '\r\n' : '\n';
  const parts = text.split(br);
  const final = text.endsWith(br);
  if (final) parts.pop();
  return { br, lines: parts, final };
}

function fromModel(m: FileModel): string {
  if (m.lines.length === 0) return '';
  return m.lines.join(m.br) + (m.final ? m.br : '');
}

const leadingWs = (s: string): string => (/^[ \t]*/.exec(s) ?? [''])[0]!;
const rtrimWs = (s: string): string => s.replace(/[ \t]+$/, '');
const trimWs = (s: string): string => s.replace(/^[ \t]+/, '').replace(/[ \t]+$/, '');

/** Positions where O occurs in lines at one tier, at or after `from`. */
function tierAt(O: string[], lines: string[], tier: number, from: number): number[] {
  const out: number[] = [];
  if (O.length === 0) return out;
  for (let i = from; i + O.length <= lines.length; i++) {
    let ok = true;
    for (let j = 0; j < O.length; j++) {
      const a = O[j]!;
      const b = lines[i + j]!;
      if (tier === 0 ? a !== b : tier === 1 ? rtrimWs(a) !== rtrimWs(b) : trimWs(a) !== trimWs(b)) {
        ok = false;
        break;
      }
    }
    if (ok) out.push(i);
  }
  return out;
}

/** The first tier (0, then 1, then 2; 0 only with --strict) at which O occurs. */
function findTiered(O: string[], lines: string[], strict: boolean, from = 0): { tier: number; pos: number[] } | null {
  for (const t of strict ? [0] : [0, 1, 2]) {
    const pos = tierAt(O, lines, t, from);
    if (pos.length > 0) return { tier: t, pos };
  }
  return null;
}

/**
 * Replace lines[at .. at+O.length) with N. ctx[j] is the index into O of the context
 * line N[j] repeats, or -1 for a genuinely new line. Context lines are written as
 * the file's own matched lines; at tier 2 the other lines of N are re-indented from
 * O's first line's leading whitespace to the file's first matched line's.
 */
function spliceLines(model: FileModel, at: number, O: string[], N: string[], ctx: number[], tier: number): FileModel {
  const out: string[] = [];
  const F = O.length > 0 ? leadingWs(model.lines[at]!) : '';
  const P = O.length > 0 ? leadingWs(O[0]!) : '';
  for (let j = 0; j < N.length; j++) {
    if (ctx[j]! >= 0) {
      out.push(model.lines[at + ctx[j]!]!);
      continue;
    }
    let line = N[j]!;
    if (tier === 2 && O.length > 0 && line !== '' && !/^[ \t]+$/.test(line) && line.startsWith(P)) {
      line = F + line.slice(P.length);
    }
    out.push(line);
  }
  return { br: model.br, final: model.final, lines: [...model.lines.slice(0, at), ...out, ...model.lines.slice(at + O.length)] };
}

/** One body line of a unified hunk or a V4A chunk: context, old, new, or backslash. */
interface BodyLine {
  k: ' ' | '-' | '+' | '\\';
  t: string;
}

/** Split a hunk/chunk body into old lines O, new lines N, and the context map for N. */
function bodyON(body: BodyLine[]): { O: string[]; N: string[]; ctx: number[] } {
  const O: string[] = [];
  const N: string[] = [];
  const ctx: number[] = [];
  for (const b of body) {
    if (b.k === ' ') {
      ctx.push(O.length);
      O.push(b.t);
      N.push(b.t);
    } else if (b.k === '-') {
      O.push(b.t);
    } else if (b.k === '+') {
      ctx.push(-1);
      N.push(b.t);
    }
  }
  return { O, N, ctx };
}

/** Whether a `\` line follows the body's last context or `+` line. */
function backslashAfterLast(body: BodyLine[]): boolean {
  let last = -1;
  for (let i = 0; i < body.length; i++) {
    if (body[i]!.k === ' ' || body[i]!.k === '+') last = i;
  }
  return last >= 0 && body[last + 1] !== undefined && body[last + 1]!.k === '\\';
}

function detect(lines: string[]): string | null {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const c = lines[first]!.trimStart()[0];
  if (c === '[' || c === '{') return 'json';
  if (lines.some((l) => l.trimEnd() === '<<<<<<< SEARCH')) return 'search-replace';
  if (lines.some((l, i) => l.startsWith('diff --git ') || (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')))) return 'unified';
  return null;
}

// ---------------------------------------------------------------------------
// Unified diffs (SPEC.md section 6)
// ---------------------------------------------------------------------------

interface UniHunk {
  a: number;
  body: BodyLine[];
}

interface UniSection {
  oldPath: string | null;
  newPath: string | null;
  kind: 'create' | 'delete' | 'move' | 'change';
  hunks: UniHunk[];
}

/** The path after `--- ` or `+++ `: text after four characters, up to the first tab, trimmed, a/ or b/ removed. */
function headerPath(line: string): string | null {
  let p = line.slice(4);
  const tab = p.indexOf('\t');
  if (tab >= 0) p = p.slice(0, tab);
  p = p.trim();
  if (p === '/dev/null') return null;
  if (p.startsWith('a/') || p.startsWith('b/')) p = p.slice(2);
  return p;
}

const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$/;
const DIFF_RE = /^diff --git a\/(\S+) b\/(\S+)\s*$/;

function parseUnified(lines: string[], strict: boolean): Edit[] {
  interface Draft {
    diffOld?: string;
    diffNew?: string;
    pairOld?: string | null;
    pairNew?: string | null;
    renameFrom?: string;
    renameTo?: string;
    newFile: boolean;
    delFile: boolean;
    hunks: UniHunk[];
  }
  const drafts: Draft[] = [];
  let cur: Draft | null = null;
  let inHeader = false;
  let i = 0;
  const newDraft = (d: Draft): Draft => {
    drafts.push(d);
    cur = d;
    inHeader = true;
    return d;
  };
  while (i < lines.length) {
    const line = lines[i]!;
    const next = lines[i + 1];
    if (line.startsWith('diff --git ')) {
      const m = DIFF_RE.exec(line);
      if (m === null) throw new Exit2(`error: parse ${i + 1}`);
      newDraft({ diffOld: m[1]!, diffNew: m[2]!, newFile: false, delFile: false, hunks: [] });
      i++;
      continue;
    }
    if (line.startsWith('--- ') && next !== undefined && next.startsWith('+++ ')) {
      if (cur !== null && inHeader && cur.pairOld === undefined) {
        cur.pairOld = headerPath(line);
        cur.pairNew = headerPath(next);
        i += 2;
        continue;
      }
      newDraft({ pairOld: headerPath(line), pairNew: headerPath(next), newFile: false, delFile: false, hunks: [] });
      i += 2;
      continue;
    }
    if (line.startsWith('--- ') && cur !== null && inHeader) {
      throw new Exit2(`error: parse ${i + 2}`);
    }
    if (cur === null) {
      i++;
      continue;
    }
    if (line.startsWith('@@')) {
      const m = HUNK_RE.exec(line);
      if (m === null) throw new Exit2(`error: parse ${i + 1}`);
      inHeader = false;
      const body: BodyLine[] = [];
      let j = i + 1;
      while (j < lines.length) {
        const bl = lines[j]!;
        if (bl.startsWith('diff --git ')) break;
        if (bl.startsWith('--- ') && (lines[j + 1] ?? '').startsWith('+++ ')) break;
        if (bl.startsWith('@@')) break;
        if (bl === '') {
          body.push({ k: ' ', t: '' });
          j++;
          continue;
        }
        const c = bl[0]!;
        if (c === ' ' || c === '-' || c === '+' || c === '\\') {
          body.push({ k: c, t: bl.slice(1) });
          j++;
          continue;
        }
        break;
      }
      while (body.length > 0) {
        const lastB = body[body.length - 1]!;
        if (lastB.k === ' ' && lastB.t === '') body.pop();
        else break;
      }
      cur.hunks.push({ a: Number(m[1]), body });
      i = j;
      continue;
    }
    if (inHeader) {
      if (line.startsWith('rename from ')) cur.renameFrom = line.slice(12).trim();
      else if (line.startsWith('rename to ')) cur.renameTo = line.slice(10).trim();
      else if (line.startsWith('new file mode')) cur.newFile = true;
      else if (line.startsWith('deleted file mode')) cur.delFile = true;
      i++;
      continue;
    }
    i++;
  }
  return drafts.map((d) => {
    const hasPair = d.pairOld !== undefined;
    const oldPath = hasPair ? d.pairOld! : (d.renameFrom ?? d.diffOld ?? null);
    const newPath = hasPair ? d.pairNew! : (d.renameTo ?? d.diffNew ?? null);
    let kind: UniSection['kind'];
    if (d.newFile) kind = 'create';
    else if (d.delFile) kind = 'delete';
    else if (oldPath === null && newPath !== null) kind = 'create';
    else if (newPath === null && oldPath !== null) kind = 'delete';
    else if (oldPath !== null && newPath !== null && oldPath !== newPath) kind = 'move';
    else kind = 'change';
    const hunks = d.hunks;
    return (tree: Tree): void => {
      if (kind === 'change' && oldPath === null && newPath === null) return;
      if (oldPath !== null && !safe(oldPath)) throw new Fail('unsafe', oldPath);
      if (newPath !== null && newPath !== oldPath && !safe(newPath)) throw new Fail('unsafe', newPath);
      if (kind === 'create') {
        if (tree.get(newPath!) !== null) throw new Fail('exists', newPath!);
        const added: string[] = [];
        for (const h of hunks) for (const b of h.body) if (b.k === '+') added.push(b.t);
        let noBreak = false;
        for (let h = hunks.length - 1; h >= 0; h--) {
          if (hunks[h]!.body.some((b) => b.k === '+')) {
            noBreak = backslashAfterLast(hunks[h]!.body);
            break;
          }
        }
        tree.now.set(newPath!, added.length === 0 ? '' : added.join('\n') + (noBreak ? '' : '\n'));
        return;
      }
      if (kind === 'delete') {
        if (tree.get(oldPath!) === null) throw new Fail('missing', oldPath!);
        tree.now.set(oldPath!, null);
        return;
      }
      const src = tree.get(oldPath!);
      if (src === null) throw new Fail('missing', oldPath!);
      if (kind === 'move' && tree.get(newPath!) !== null) throw new Fail('exists', newPath!);
      let model = toModel(src);
      let d = 0;
      for (const h of hunks) {
        const { O, N, ctx } = bodyON(h.body);
        if (O.length === 0) {
          const at = Math.max(0, Math.min(h.a + d, model.lines.length));
          const atEnd = at === model.lines.length;
          model = spliceLines(model, at, O, N, ctx, 0);
          if (atEnd) model.final = !backslashAfterLast(h.body);
          d += N.length;
          continue;
        }
        const found = findTiered(O, model.lines, strict);
        if (found === null) throw new Fail('no-match', oldPath!);
        const target = h.a - 1 + d;
        let at = found.pos[0]!;
        let best = Math.abs(at - target);
        for (const p of found.pos) {
          const dist = Math.abs(p - target);
          if (dist < best) {
            at = p;
            best = dist;
          }
        }
        const touchesEnd = at + O.length === model.lines.length;
        model = spliceLines(model, at, O, N, ctx, found.tier);
        if (touchesEnd) model.final = !backslashAfterLast(h.body);
        d += N.length - O.length;
      }
      const text = fromModel(model);
      if (kind === 'move') {
        tree.now.set(oldPath!, null);
        tree.now.set(newPath!, text);
        tree.moves.push([oldPath!, newPath!]);
      } else {
        tree.now.set(oldPath!, text);
      }
    };
  });
}

// ---------------------------------------------------------------------------
// SEARCH/REPLACE blocks (SPEC.md section 7)
// ---------------------------------------------------------------------------

function parseSearchReplace(lines: string[], strict: boolean): Edit[] {
  const edits: Edit[] = [];
  let prevEnd = 0;
  let prevPath: string | null = null;
  let i = 0;
  while (i < lines.length) {
    if (lines[i]!.trimEnd() !== '<<<<<<< SEARCH') {
      i++;
      continue;
    }
    const searchLine = i;
    let path: string | null = null;
    for (let j = searchLine - 1; j >= prevEnd; j--) {
      const l = lines[j]!;
      if (blank(l)) continue;
      if (l.trimStart().startsWith('```')) continue;
      path = l.trim();
      break;
    }
    if (path === null) path = prevPath;
    if (path === null) throw new Exit2(`error: parse ${searchLine + 1}`);
    let eq = -1;
    let rep = -1;
    for (let j = searchLine + 1; j < lines.length; j++) {
      if (lines[j]!.trimEnd() === '=======') {
        eq = j;
        break;
      }
    }
    if (eq >= 0) {
      for (let j = eq + 1; j < lines.length; j++) {
        if (lines[j]!.trimEnd() === '>>>>>>> REPLACE') {
          rep = j;
          break;
        }
      }
    }
    if (eq < 0 || rep < 0) throw new Exit2(`error: parse ${searchLine + 1}`);
    const oldLines = lines.slice(searchLine + 1, eq);
    const newLines = lines.slice(eq + 1, rep);
    edits.push(((p, oldL, newL) => (tree: Tree): void => {
      if (!safe(p)) throw new Fail('unsafe', p);
      if (oldL.length === 0) {
        if (tree.get(p) !== null) throw new Fail('exists', p);
        tree.now.set(p, newL.length === 0 ? '' : newL.join('\n') + '\n');
        return;
      }
      const cur = tree.get(p);
      if (cur === null) throw new Fail('missing', p);
      const model = toModel(cur);
      const found = findTiered(oldL, model.lines, strict);
      if (found === null) throw new Fail('no-match', p);
      if (found.pos.length > 1) throw new Fail('ambiguous', p);
      const out = spliceLines(model, found.pos[0]!, oldL, newL, newL.map(() => -1), found.tier);
      tree.now.set(p, fromModel(out));
    })(path, oldLines, newLines));
    prevPath = path;
    prevEnd = rep + 1;
    i = rep + 1;
  }
  return edits;
}

// ---------------------------------------------------------------------------
// V4A patches (SPEC.md section 8)
// ---------------------------------------------------------------------------

function parseV4a(lines: string[], strict: boolean): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  if (lines[first]!.trim() !== '*** Begin Patch') throw new Exit2(`error: parse ${first + 1}`);
  let last = -1;
  for (let k = lines.length - 1; k >= 0; k--) {
    if (!blank(lines[k]!)) {
      last = k;
      break;
    }
  }
  if (lines[last]!.trim() !== '*** End Patch') throw new Exit2(`error: parse ${lines.length + 1}`);
  const edits: Edit[] = [];
  const isSectionStart = (l: string): boolean =>
    l.startsWith('*** Add File: ') || l.startsWith('*** Delete File: ') || l.startsWith('*** Update File: ');
  let i = first + 1;
  while (i < last) {
    const line = lines[i]!;
    if (blank(line)) {
      i++;
      continue;
    }
    if (!isSectionStart(line)) throw new Exit2(`error: parse ${i + 1}`);
    const headLine = i + 1;
    const kind = line.startsWith('*** Add File: ') ? 'add' : line.startsWith('*** Delete File: ') ? 'delete' : 'update';
    const prefixLen = kind === 'add' ? 14 : 17;
    const path = line.slice(prefixLen).trim();
    const raw: Array<{ text: string; num: number }> = [];
    let j = i + 1;
    while (j < last && !isSectionStart(lines[j]!)) {
      raw.push({ text: lines[j]!, num: j + 1 });
      j++;
    }
    while (raw.length > 0 && blank(raw[raw.length - 1]!.text)) raw.pop();
    if (kind === 'add') {
      const fileLines: string[] = [];
      for (const r of raw) {
        if (r.text === '') fileLines.push('');
        else if (r.text.startsWith('+')) fileLines.push(r.text.slice(1));
        else throw new Exit2(`error: parse ${r.num}`);
      }
      edits.push(((p, fl) => (tree: Tree): void => {
        if (!safe(p)) throw new Fail('unsafe', p);
        if (tree.get(p) !== null) throw new Fail('exists', p);
        tree.now.set(p, fl.length === 0 ? '' : fl.join('\n') + '\n');
      })(path, fileLines));
    } else if (kind === 'delete') {
      for (const r of raw) throw new Exit2(`error: parse ${r.num}`);
      edits.push(((p) => (tree: Tree): void => {
        if (!safe(p)) throw new Fail('unsafe', p);
        if (tree.get(p) === null) throw new Fail('missing', p);
        tree.now.set(p, null);
      })(path));
    } else {
      let k = 0;
      let moveTo: string | undefined;
      if (k < raw.length && raw[k]!.text.startsWith('*** Move to: ')) {
        moveTo = raw[k]!.text.slice(13).trim();
        k++;
      }
      interface VChunk {
        anchor: string | null;
        eof: boolean;
        O: string[];
        N: string[];
        ctx: number[];
      }
      const chunks: VChunk[] = [];
      const isBody = (t: string): boolean => t === '' || t[0] === ' ' || t[0] === '-' || t[0] === '+';
      while (k < raw.length) {
        const l = raw[k]!;
        let anchor: string | null = null;
        if (l.text.startsWith('@@')) {
          anchor = l.text.slice(2).trim();
          if (anchor === '') anchor = null;
          k++;
        } else if (chunks.length === 0 && isBody(l.text)) {
          // First chunk may start at a body line; anchor stays null.
        } else {
          throw new Exit2(`error: parse ${l.num}`);
        }
        const body: BodyLine[] = [];
        let eof = false;
        while (k < raw.length) {
          const b = raw[k]!;
          if (b.text === '*** End of File') {
            eof = true;
            k++;
            break;
          }
          if (b.text.startsWith('@@')) break;
          if (isBody(b.text)) {
            body.push(b.text === '' ? { k: ' ', t: '' } : { k: b.text[0] as ' ' | '-' | '+', t: b.text.slice(1) });
            k++;
            continue;
          }
          throw new Exit2(`error: parse ${b.num}`);
        }
        const on = bodyON(body);
        chunks.push({ anchor, eof, O: on.O, N: on.N, ctx: on.ctx });
      }
      if (chunks.length === 0 && moveTo === undefined) throw new Exit2(`error: parse ${headLine}`);
      edits.push(((p, mt, chs) => (tree: Tree): void => {
        if (!safe(p)) throw new Fail('unsafe', p);
        if (mt !== undefined && !safe(mt)) throw new Fail('unsafe', mt);
        const cur = tree.get(p);
        if (cur === null) throw new Fail('missing', p);
        if (mt !== undefined && tree.get(mt) !== null) throw new Fail('exists', mt);
        let model = toModel(cur);
        let cursor = 0;
        for (const ch of chs) {
          if (ch.anchor !== null) {
            const a = findTiered([ch.anchor], model.lines, strict, cursor);
            if (a === null) throw new Fail('no-match', p);
            cursor = a.pos[0]! + 1;
          }
          if (ch.O.length === 0) {
            const at = ch.eof ? model.lines.length : cursor;
            model = spliceLines(model, at, ch.O, ch.N, ch.ctx, 0);
            cursor = at + ch.N.length;
            continue;
          }
          const found = findTiered(ch.O, model.lines, strict, cursor);
          if (found === null) throw new Fail('no-match', p);
          let at = found.pos[0]!;
          if (ch.eof) {
            const end = found.pos.filter((q) => q + ch.O.length === model.lines.length);
            if (end.length === 0) throw new Fail('no-match', p);
            at = end[0]!;
          }
          model = spliceLines(model, at, ch.O, ch.N, ch.ctx, found.tier);
          cursor = at + ch.N.length;
        }
        const text = fromModel(model);
        if (mt !== undefined) {
          tree.now.set(p, null);
          tree.now.set(mt, text);
          tree.moves.push([p, mt]);
        } else {
          tree.now.set(p, text);
        }
      })(path, moveTo, chunks));
    }
    i = j;
  }
  return edits;
}

// ---------------------------------------------------------------------------
// JSON edits (SPEC.md section 9)
// ---------------------------------------------------------------------------

/** One JSON edit object (SPEC.md section 9), or null when it is not one. */
function jsonEdit(v: unknown): Edit | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const p = o.path;
  if (typeof p !== 'string') return null;
  const exists = (tree: Tree): string => {
    if (!safe(p)) throw new Fail('unsafe', p);
    const t = tree.get(p);
    if (t === null) throw new Fail('missing', p);
    return t;
  };
  if (o.command === 'create' && typeof o.file_text === 'string') {
    const text = o.file_text;
    return (tree) => {
      if (!safe(p)) throw new Fail('unsafe', p);
      if (tree.get(p) !== null) throw new Fail('exists', p);
      tree.now.set(p, text);
    };
  }
  if (o.command === 'str_replace' && typeof o.old_str === 'string' && o.old_str !== '' && (o.new_str === undefined || typeof o.new_str === 'string')) {
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const crlf = nl > 0 && before[nl - 1] === '\r';
      const lf = (s: string): string => (crlf ? s.replaceAll('\r\n', '\n') : s);
      const t = lf(before);
      const old = lf(oldStr);
      let count = 0;
      for (let k = t.indexOf(old); k >= 0; k = t.indexOf(old, k + 1)) count += 1;
      if (count === 0) throw new Fail('no-match', p);
      if (count > 1) throw new Fail('ambiguous', p);
      const at = t.indexOf(old);
      const out = t.slice(0, at) + lf(newStr) + t.slice(at + old.length);
      tree.now.set(p, crlf ? out.replaceAll('\n', '\r\n') : out);
    };
  }
  if (o.command === 'insert' && typeof o.insert_line === 'number' && Number.isInteger(o.insert_line) && typeof o.new_str === 'string') {
    const line = o.insert_line;
    const add = patchLines(o.new_str);
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const br = nl > 0 && before[nl - 1] === '\r' ? '\r\n' : '\n';
      const lines = before === '' ? [] : before.split(br);
      const final = before !== '' && before.endsWith(br);
      if (final) lines.pop();
      if (line < 0 || line > lines.length) throw new Fail('no-match', p);
      lines.splice(line, 0, ...add);
      tree.now.set(p, lines.length === 0 ? '' : lines.join(br) + (final ? br : ''));
    };
  }
  return null;
}

function jsonEdits(text: string, lines: string[]): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  const edits: Edit[] = [];
  if (lines[first]!.trimStart().startsWith('[')) {
    let v: unknown;
    try {
      v = JSON.parse(text);
    } catch {
      throw new Exit2(`error: parse ${first + 1}`);
    }
    if (!Array.isArray(v)) throw new Exit2(`error: parse ${first + 1}`);
    for (const x of v) {
      const e = jsonEdit(x);
      if (e === null) throw new Exit2(`error: parse ${first + 1}`);
      edits.push(e);
    }
    return edits;
  }
  lines.forEach((l, i) => {
    if (blank(l)) return;
    let e: Edit | null = null;
    try {
      e = jsonEdit(JSON.parse(l));
    } catch {
      e = null;
    }
    if (e === null) throw new Exit2(`error: parse ${i + 1}`);
    edits.push(e);
  });
  return edits;
}

// ---------------------------------------------------------------------------
// Entry point (SPEC.md sections 1, 2 and 10)
// ---------------------------------------------------------------------------

function run(args: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let check = false;
  let strict = false;
  const pos: string[] = [];
  let ended = false;
  for (const a of args) {
    if (!ended && a === '--') ended = true;
    else if (!ended && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null && FORMATS.includes(a.slice(9))) format = a.slice(9);
      else if (a === '--check' && !check) check = true;
      else if (a === '--strict' && !strict) strict = true;
      else return { code: 2, out: ['error: usage'] };
    } else pos.push(a);
  }
  if (pos.length > 1) return { code: 2, out: ['error: usage'] };
  const file = pos.length === 0 || pos[0] === '-' ? null : pos[0]!;
  let text = '';
  try {
    text = readFileSync(file ?? 0, 'utf8');
  } catch {
    if (file !== null) return { code: 2, out: ['error: no-patch'] };
  }
  const lines = patchLines(text);
  const kind = format === null || format === 'auto' ? detect(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };
  let edits: Edit[];
  try {
    edits =
      kind === 'json'
        ? jsonEdits(text, lines)
        : kind === 'unified'
          ? parseUnified(lines, strict)
          : kind === 'search-replace'
            ? parseSearchReplace(lines, strict)
            : parseV4a(lines, strict);
    if (edits.length === 0) throw new Exit2('error: parse 1');
  } catch (e) {
    if (e instanceof Exit2) return { code: 2, out: [e.line] };
    throw e;
  }
  const tree = new Tree();
  const fails: Fail[] = [];
  for (const edit of edits) {
    const before = new Map(tree.now);
    const movesBefore = tree.moves.length;
    try {
      edit(tree);
    } catch (e) {
      if (!(e instanceof Fail)) throw e;
      tree.now.clear();
      for (const [k, v] of before) tree.now.set(k, v);
      tree.moves.length = movesBefore;
      fails.push(e);
    }
  }
  if (fails.length > 0) return { code: 1, out: [...fails.map((f) => `error: ${f.code} ${f.path}`), `failed: ${fails.length} of ${edits.length} edits`] };
  const byteOrder = (a: string, b: string): number => Buffer.compare(Buffer.from(a, 'utf8'), Buffer.from(b, 'utf8'));
  const rows: Array<{ key: string; line: string }> = [];
  const rOld = new Set<string>();
  const rNew = new Set<string>();
  for (const [o, n] of tree.moves) {
    if (tree.original(o) !== null && tree.get(o) === null && tree.original(n) === null && tree.get(n) !== null) {
      rows.push({ key: n, line: `R ${o} -> ${n}` });
      rOld.add(o);
      rNew.add(n);
    }
  }
  for (const p of [...tree.now.keys()].sort(byteOrder)) {
    if (rOld.has(p) || rNew.has(p)) continue;
    const was = tree.original(p);
    const now = tree.now.get(p)!;
    if (was === now) continue;
    rows.push({ key: p, line: `${was === null ? 'A' : now === null ? 'D' : 'M'} ${p}` });
  }
  if (!check) {
    for (const p of [...tree.now.keys()].sort(byteOrder)) {
      const was = tree.original(p);
      const now = tree.now.get(p)!;
      if (was === now) continue;
      if (now === null) rmSync(p, { force: true });
      else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  rows.sort((a, b) => byteOrder(a.key, b.key));
  const outLines = rows.map((r) => r.line);
  return { code: 0, out: [...outLines, `ok: ${outLines.length} ${outLines.length === 1 ? 'file' : 'files'}`] };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = run(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}
