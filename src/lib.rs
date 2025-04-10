use burn::{
    module::{AutodiffModule, ModuleVisitor, ParamId},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::{Backend, Tensor},
    tensor::backend::AutodiffBackend,
};

use core::f64;

use linfa_nn::{BallTree, distance::L1Dist, NearestNeighbour};
use ndarray::{Array, array};

use burn::tensor::ElementConversion;

/// A Burn Module must implement a `pdf` on data function
pub trait ModelTrait<B: AutodiffBackend>: AutodiffModule<B> {
    fn pdf(&self, data: &Tensor<B, 1>) -> Tensor<B, 1>;
    fn f_log(&self) -> Tensor<B, 1>;
}

const EXP_EULER_GAMMA_1: f64 = 0.6552199258161035685581240413758575978_f64;

/// This `forward` function calculates the estimate for
/// Kullback-Leibler divergence. At the moment, it assumes that
/// the volume balls do not depend on the parameters of the model.
pub fn forward<B: AutodiffBackend, M: ModelTrait<B>>(
    model: &M,
    data: &Tensor<B, 1>,
    balls: &Tensor<B, 1>
) -> Tensor<B, 1> {
    let pdf = model.pdf(data);
    let factor: f64 = (data.shape().dims[0] as f64) * EXP_EULER_GAMMA_1;
    let sum_terms = balls.clone().mul_scalar(factor).log().mul(balls.clone()).mul(pdf);
    model.f_log() - sum_terms.sum()
}

fn calculate_balls<B: Backend>(data: &Vec<f64>, split: bool, device: &B::Device) -> (Tensor<B, 1>, Tensor<B, 1>) {

    // considered that the sample could be split to ensure i.i.d terms in the sum
    // but there were no apparent benefits; leaving this legacy in case need to investigate
    // again
    let num = data.len();
    let data1 = if split { &data[0..(num / 2)] } else { &data[..] };  // slice used for calculate volume to nearest
    let data2 = if split { &data[(num / 2)..] } else { &data[..] };   // slice used to iterate points

    let algo = BallTree::new();
    let arr = Array::from_shape_vec([data1.len(), 1], data1.to_vec()).unwrap();
    let arr = arr.view();
    let nn_index = algo.from_batch(&arr, L1Dist).unwrap();
    let pos_nearest = 1;

    let radii: Vec<f64> = data2.iter()
        .map(|pt: &f64| (nn_index.k_nearest((array![*pt]).view(), pos_nearest + 1).unwrap(), pt))
        .map(|resp: (Vec<(ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 1]>>, usize)>, &f64)|
                    (resp.1 - resp.0[pos_nearest].0[0]).abs())  // distance to nearest neighbour
        .map(|v: f64| v * 2.0)                        // ball volume in dimension 1
        .collect();

    (
        Tensor::from_data(data2, device),
        Tensor::from_data(radii.as_slice(), device)
    )
}

pub struct TrainingConfig {
    pub num_runs: usize,
    pub lr: f64,
    pub config_optimizer: AdamConfig,
}

struct GradientCheck<'a, B: AutodiffBackend> {
    epsilon: f64,
    is_less: bool,
    grads: &'a B::Gradients,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradientCheck<'_, B> {
    fn visit_float<const D: usize>(&mut self, _id: ParamId, tensor: &Tensor<B, D>) {
        if self.is_less {
            let val: f64 = tensor.grad(&self.grads).unwrap().into_scalar().elem();
            self.is_less = val.abs() < self.epsilon;
        }
    }
}

pub fn run<B: AutodiffBackend, M: ModelTrait<B>>(
    mut model: M,
    vec: Vec<f64>,
    split: bool,
    device: B::Device,
) -> (usize, M) {

    let config = TrainingConfig {
        num_runs: 1000,
        lr: 0.01,
        config_optimizer: AdamConfig::new(),
    };

    let balls = calculate_balls::<B>(&vec, split, &device);

    let mut optimizer = config.config_optimizer.init::<B, M>();
    let epsilon: f64 = 0.000001;

    let mut ix = 1;
    while ix <= config.num_runs {

        let kl_hat = forward(&model, &balls.0, &balls.1);

        let grads = kl_hat.backward();

        let is_less = {
            let mut grad_check = GradientCheck {
                epsilon,
                is_less: true,
                grads: &grads,
            };
            model.visit(&mut grad_check);
            grad_check.is_less
        };
        
        let grads_container = GradientsParams::from_grads(grads, &model);

        model = optimizer.step(config.lr, model, grads_container);

        let kl_hat_val: f64 = kl_hat.into_scalar().elem::<f64>();

        if ix % 10 == 0 {
            println!("KL Hat: {} ({})", kl_hat_val, ix);
        }
        if is_less {
            break;
        }
        ix += 1;
    }
    (ix, model)
}
