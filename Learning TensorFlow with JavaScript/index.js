// shape our tenser with 4 rows of 2 columns
const shape = [4, 2]
// const data = tf.tensor([[4, 6], [5, 9], [12, 25], [1, 57]])

// feed data into the tensor
const data = tf.tensor([4, 6, 5, 9, 12, 25, 1, 57], shape)

// set variables with zeros method
const data2 = tf.variable(tf.zeros([8]));



// print the data
data.print()
data2.print()

// this is where we assign new values with 1 dimension
data2.assign(tf.tensor1d([4, 25, 5, 56, 3, 45, 4, 6]));
data2.print()

// create 2 new 1 dimensional tensors
const data3 = tf.tensor1d([4, 5, 6, 8]);
const data4 = tf.tensor1d([40, 50, 60, 80]);

// prints
data3.print();
data4.print();

// adds and multiplies and prints
data3.add(data4).print();
data3.mul(data4).print();

// second part

// define my model
function simpleAdd(input1, input2) {
    // tidy is used to free up GPU memory once the functions returns
    return tf.tidy(() => {
        const x1 = input1;
        const x2 = input2;
        const y = x1.add(x2);
        return y;
    });
}

// new 1 dimensional tensors/arrays
const data5 = tf.tensor1d([4, 6, 5, 9]);
const data6 = tf.tensor1d([40, 60, 50, 90]);

// using the model to do input to output
const result = simpleAdd(data5, data6);

// printing result
result.print();


// sequential model
const model = tf.sequential();


// add first layer
model.add(
    tf.layers.simpleRNN({
        // only needed first layer
        inputShape: [20, 4],
        // the number of units or neurons
        units: 20,
        // weight
        recurrentInitializer: 'GlorotNormal',
    })
);


