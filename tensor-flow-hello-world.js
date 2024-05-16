const tf = require("@tensorflow/tfjs");

/*
    step 1: model creations
    step 2: data prepare
    step 3: model training
    step 4: model prediction
*/

async function run() {
  console.log("Inside run");

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

  const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
  const ys = tf.tensor2d([[2], [5], [8], [11]], [4, 1]);

  await model.fit(xs, ys, { epochs: 3000 });

  const input = tf.tensor2d([[5]], [1, 1]);

  const output = model.predict(input);
  output.print();
}

run();
