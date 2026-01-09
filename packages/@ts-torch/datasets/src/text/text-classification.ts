/**
 * Text classification datasets
 */

import { BaseDataset } from '../dataset.js';

/**
 * Simple text classification dataset
 */
export class TextClassificationDataset extends BaseDataset<[string, number]> {
  constructor(
    private texts: string[],
    private labels: number[]
  ) {
    super();

    if (texts.length !== labels.length) {
      throw new Error('Texts and labels must have the same length');
    }
  }

  getItem(index: number): [string, number] {
    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`);
    }

    return [this.texts[index]!, this.labels[index]!];
  }

  get length(): number {
    return this.texts.length;
  }
}

/**
 * CSV text classification dataset
 */
export class CSVTextDataset extends BaseDataset<[string, number]> {
  private data: Array<[string, number]> = [];

  constructor(
    _csvPath: string,
    _textColumn: string = 'text',
    _labelColumn: string = 'label'
  ) {
    super();
  }

  async init(): Promise<void> {
    // TODO: Load and parse CSV file
    // const csv = await readFile(this.csvPath, 'utf-8');
    // const rows = parseCSV(csv);
    // this.data = rows.map(row => [row[this.textColumn], parseInt(row[this.labelColumn])]);
  }

  getItem(index: number): [string, number] {
    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`);
    }

    return this.data[index]!;
  }

  get length(): number {
    return this.data.length;
  }
}
