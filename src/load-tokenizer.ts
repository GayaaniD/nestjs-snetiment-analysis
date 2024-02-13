import * as fs from 'fs';
import { load } from 'node-pickle';

export function loadPickle(filePath: string): any {
  // Read the pickle file
  const dataBuffer = fs.readFileSync(filePath);

  // Deserialize the pickle file
  const deserializedData = load(dataBuffer);

  return deserializedData;
}
