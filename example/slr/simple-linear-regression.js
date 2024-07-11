const tf = require("@tensorflow/tfjs");
const path = require("path");
/*
    - Data import
    - Model configure
    - Model train
    - Prediction
*/
async function run() {
  console.log("Inside run");
  const csvFilePath = path.resolve(__dirname, "house-prices3.csv");
  console.log(csvFilePath);
  const columnConfig = {
    SquareFootage: { dtype: "int32" },
    Price: { dtype: "int32" },
  };

  const csvContent = tf.data.csv(`file://${csvFilePath}`, columnConfig);
  console.log(csvContent);

  const xValues = [];
  const yValues = [];
  await csvContent.forEachAsync((row) => {
    xValues.push([row.SquareFootage]);
    yValues.push([row.Price]);
  });

  const xs = tf.tensor(xValues);
  const ys = tf.tensor(yValues);

  console.log(xs, ys);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({ optimizer: "adam", loss: "meanAbsoluteError" }); // 30004
  //model.compile({ optimizer: "adam", loss: "meanSquaredError" }); // 29777
  // model.compile({ optimizer: "sgd", loss: "meanSquaredError" }); // Nan
  //model.compile({ optimizer: "sgd", loss: "meanAbsoluteError" }); // 51477

  await model.fit(xs, ys, { epochs: 30000 });

  const input = tf.tensor([[1500]], [1, 1]);

  const output = model.predict(input);

  output.print();
}

run();
