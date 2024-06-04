const tf = require("@tensorflow/tfjs-node");
const path = require("path");

/*
    step 1: model creations
    step 2: data prepare
    step 3: model training
    step 4: model prediction
*/

async function run() {
  console.log("Inside run");

  const csvFilePath =
    "https://ashish-sal-test.s3.eu-west-1.amazonaws.com/sample_data.csv";
  const xValues = [];
  const yValues = [];

  console.log(csvFilePath);

  const columnConfig = {
    number1: { dtype: "int32" },
    number2: { dtype: "int32" },
  };
  const csvDataSet = tf.data.csv(csvFilePath, columnConfig);

  await csvDataSet.forEachAsync((row) => {
    if (!Number.isInteger(row.number1) || !Number.isInteger(row.number2)) {
      throw new Error("number1 and number2 should be integers");
    }

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
