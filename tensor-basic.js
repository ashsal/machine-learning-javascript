const tf = require("@tensorflow/tfjs");

const oneDimension = [1, 2, 3];
console.log(oneDimension);

const twoDimension = [
  [1, 2, 3],
  [4, 5, 6],
];

console.log(twoDimension);

const threeDimension = [
  [
    [1, 2, 3],
    [4, 5, 6],
  ],
  [
    [7, 8, 9],
    [10, 11, 12],
  ],
];
console.log(threeDimension);

async function run() {
  console.log("Inside run");
  const scalar = tf.scalar(8);
  scalar.print();

  const vector = tf.tensor([1, 2, 3]);
  vector.print();
  console.log("vector.rank", vector.rank);
  console.log("vector.shape", vector.shape);

  const vector2 = tf.tensor1d([1, 2, 3]);
  vector2.print();
  console.log("vector2.rank", vector2.rank);
  console.log("vector2.shape", vector2.shape);

  const matrix = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  matrix.print();
  console.log("matrix.rank", matrix.rank);
  console.log("matrix.shape", matrix.shape);

  const matrix1 = tf.tensor2d([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  matrix1.print();
  console.log("matrix1.rank", matrix1.rank);
  console.log("matrix1.shape", matrix1.shape);

  const threeDMatrix = tf.tensor([
    [
      [1, 2, 3],
      [4, 5, 6],
    ],
    [
      [7, 8, 9],
      [10, 11, 12],
    ],
  ]);

  threeDMatrix.print();
  console.log("threeDMatrix.rank", threeDMatrix.rank);
  console.log("threeDMatrix.shape", threeDMatrix.shape);

  console.log("sum is", vector.sum());
  vector.sum().print();
  console.log("vector.sum().shape", vector.sum().shape);
  console.log("vector.sum().rank", vector.sum().rank);

  vector.mean().print();

  const addResult = tf.add(vector, vector2);
  addResult.print();
  console.log("addResult.rank", addResult.rank);
  console.log("addResult.shape", addResult.shape);

  const multTensor1 = tf.tensor2d([
    [1, 2],
    [3, 4],
  ]);

  const multTensor2 = tf.tensor2d([
    [5, 6],
    [7, 8],
  ]);

  // 1 * 5 + 2 * 7 = 5 + 14 = 19
  const multResult = tf.matMul(multTensor1, multTensor2);
  multResult.print();

  multTensor2.transpose().print();

  const reshapeInput = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  reshapeInput.print();
  console.log("reshapeInput.shape", reshapeInput.shape);
  reshapeInput.reshape([3, 2]).print();

  const sum0Output = reshapeInput.sum(0);
  sum0Output.print();

  const sum1Output = reshapeInput.sum(1);
  sum1Output.print();
}

run();
