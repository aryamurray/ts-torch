/**
 * Text classification datasets
 */

import { readFile } from 'fs/promises'
import { BaseDataset } from '../dataset.js'

/**
 * Simple text classification dataset
 */
export class TextClassificationDataset extends BaseDataset<[string, number]> {
  constructor(
    private texts: string[],
    private labels: number[],
  ) {
    super()

    if (texts.length !== labels.length) {
      throw new Error('Texts and labels must have the same length')
    }
  }

  getItem(index: number): [string, number] {
    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`)
    }

    return [this.texts[index]!, this.labels[index]!]
  }

  get length(): number {
    return this.texts.length
  }
}

/**
 * CSV text classification dataset
 */
export class CSVTextDataset extends BaseDataset<[string, number]> {
  private data: Array<[string, number]> = []
  private csvPath: string
  private textColumn: string
  private labelColumn: string

  constructor(csvPath: string, textColumn: string = 'text', labelColumn: string = 'label') {
    super()
    this.csvPath = csvPath
    this.textColumn = textColumn
    this.labelColumn = labelColumn
  }

  async init(): Promise<void> {
    const csv = await readFile(this.csvPath, 'utf-8')
    const rows = parseCSV(csv)
    if (rows.length === 0) {
      this.data = []
      return
    }
    const header = rows[0]!
    const textIdx = header.indexOf(this.textColumn)
    const labelIdx = header.indexOf(this.labelColumn)
    if (textIdx === -1 || labelIdx === -1) {
      throw new Error(`CSV missing required columns: ${this.textColumn}, ${this.labelColumn}`)
    }

    this.data = rows.slice(1).map((row) => {
      const text = row[textIdx] ?? ''
      const label = Number.parseInt(row[labelIdx] ?? '0', 10)
      return [text, Number.isNaN(label) ? 0 : label]
    })
  }

  getItem(index: number): [string, number] {
    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`)
    }

    return this.data[index]!
  }

  get length(): number {
    return this.data.length
  }
}

function parseCSV(content: string): string[][] {
  const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0)
  return lines.map(parseCSVLine)
}

function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const char = line[i]!
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"'
        i++
      } else {
        inQuotes = !inQuotes
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current)
      current = ''
    } else {
      current += char
    }
  }
  result.push(current)
  return result.map((value) => value.trim())
}
