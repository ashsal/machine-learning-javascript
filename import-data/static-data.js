const tf = require("@tensorflow/tfjs-node");

/*
    step 1: model creations
    step 2: data prepare
    step 3: model training
    step 4: model prediction
*/

async function run() {
  console.log("Inside run");

  const xValues = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
  const yValues = [[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]];

  const xs = tf.tensor(xValues);
  const ys = tf.tensor(yValues);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

  await model.fit(xs, ys, { epochs: 500 }); // Training

  const input = tf.tensor2d([[15]], [1, 1]);

  const output = model.predict(input);
  output.print();
}

run();
