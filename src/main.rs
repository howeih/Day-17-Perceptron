#[macro_use(array)]
extern crate ndarray;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Zip;

fn perceptron(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let dot = x.dot(y);
    if dot >= 0f64 {
        1.0
    } else {
        0.0
    }
}

fn perceptron2(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let mut dot = x.dot(y);
    for d in dot.iter_mut() {
        if *d >= 0f64 {
            *d = 1.0;
        } else {
            *d = 0.0;
        }
    }

    dot
}

fn train(x: &Array2<f64>, y: &Array1<f64>, w: &mut Array1<f64>) -> Array1<f64> {
    let mut i: usize = 0;
    for xr in x.genrows() {
        let h = perceptron(&xr.to_owned(), &w.to_owned());
        if h != y[i] {
            let pos;
            if y[i] == 1.0 {
                pos = w.to_owned() + xr.to_owned(); // xr+w;
            } else {
                pos = w.to_owned() - xr.to_owned(); // xr+w;
            }
            for (c, z) in pos.iter().enumerate() {
                w[c] = *z;
            }
        }
        i += 1;
    }
    perceptron2(x, w)
}
fn mean(h: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let mut total = 0.;
    let mut eq_count = 0.;
    Zip::from(h).and(y).apply(|h, y| {
        total += 1.;
        if h == y {
            eq_count += 1.;
        }
    });
    eq_count / total
}

fn main() {
    let x = array![
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.],
        [-1., 1., 1.],
        [1., -1., 1.]
    ];
    let y = array![1., 1., 1., 0., 0.];
    let mut w = array![0., 0., 0.];

    for _ in 0..5 {
        let h = train(&x, &y, &mut w);
        println!("w={:?} acc={}", w, mean(&h, &y));
    }
}
