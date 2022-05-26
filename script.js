// y=2x +3
const modelo = tf.sequential();

modelo.add(tf.layers.dense({
  inputShape: 1,
  units: 1,
}));
modelo.add(tf.layers.dense({
  units: 1,
}))

modelo.compile({
  optimizer: "sgd",
  loss: "meanSquaredError",
});

const xs = tf.tensor([-1, 0, 1, 2, 3, 4], [6, 1])
const ys = tf.tensor([1, 3, 5, 7, 9, 11], [6, 1]);

modelo
  .fit(xs, ys, { epochs: 100 })
  .then(() => {
    const TensorX = tf.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const datosTensor = modelo.predict(TensorX).dataSync();
    const [...valores] = datosTensor;

    const res = valores.map((y, x) => ({ x, y }));

    console.log("Resultados", res)

    tf.dispose([xs, ys, modelo, datosTensor, TensorX]);
  });

// Gonzalez Gabriel
// Grupo 1