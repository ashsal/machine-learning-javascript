const mysql = require("mysql2/promise");
const tf = require("@tensorflow/tfjs-node");

async function run() {
  console.log("Inside run");

  const xValues = [];
  const yValues = [];

  const connection = await mysql.createConnection({
    host: "localhost",
    user: "ashish",
    password: "123456",
    database: "machine_learning",
  });

  const [results, fields] = await connection.query(
    `select * from sampla_data_1`
  );
  results.forEach((row) => {
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
