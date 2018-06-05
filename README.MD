#Day 17: Perceptron
Perceptron is a beautiful statistical learning device for classification.
<br/>
Perceptron is very simple to implement, it is an online algorithm, but what’s most important, it is a combination of mathematical function, learning algorithm and a proof of the algorithm correctness when applied on linearly separable sets.
<br/>
<br/>
data:
<pre>
    let x = array![
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.],
        [-1., 1., 1.],
        [1., -1., 1.]
    ];
    let y = array![1., 1., 1., 0., 0.];
    let mut w = array![0., 0., 0.];
</pre>
training:
<pre>
for _ in 0..5 {
    let h = train(&x, &y, &mut w);
    println!("w={:?} acc={}", w, mean(&h, &y));
}
w=[0.0, 0.0, -2.0] shape=[3], strides=[1], layout=C | F (0x3) acc=0.4
w=[1.0, 1.0, -2.0] shape=[3], strides=[1], layout=C | F (0x3) acc=0.6
w=[2.0, 1.0, -2.0] shape=[3], strides=[1], layout=C | F (0x3) acc=0.8
w=[2.0, 2.0, -1.0] shape=[3], strides=[1], layout=C | F (0x3) acc=1
w=[2.0, 2.0, -1.0] shape=[3], strides=[1], layout=C | F (0x3) acc=1
</pre>
