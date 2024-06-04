const tf = require("@tensorflow/tfjs-node");
const axios = require("axios");
/*
    step 1: model creations
    step 2: data prepare
    step 3: model training
    step 4: model prediction
*/

async function run() {
  console.log("Inside run");

  const xValues = [];
  const yValues = [];

  const mockDataResponse = await axios.get(
    "https://mocki.io/v1/3f0f13ea-772c-4c08-b89b-028bc7658668"
  );

  if (mockDataResponse.status !== 200) {
    throw new Error("Get data from json endpoint failed");
  }

  mockDataResponse.data.forEach((row) => {
    xValues.push([row.number1]);
    yValues.push([row.number2]);
  });

  console.log(xValues, yValues);

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
