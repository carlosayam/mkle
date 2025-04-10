use mkl::{run, ModelTrait};

use burn::{
    backend::{
        Autodiff,
        NdArray
    }, module::{
        Module,
        Param
    }, prelude::{
        Backend, Tensor
    }, tensor::backend::AutodiffBackend,
};
use argparse::{ArgumentParser, Store, StoreOption, StoreTrue};

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

use rand::distributions::Distribution;
use statrs::distribution::Normal;


#[derive(Module, Debug)]
pub struct NormalModel<B: Backend> {
    loc: Param<Tensor<B, 1>>,
    scale: Param<Tensor<B, 1>>,
}

/// square root of 2 \pi
const SQRT_TWO_PI: f64 = 2.50662827463100050241576528481104525300699_f64;

///  -(1/2) - 1/2 Log[2 \[Pi]], a constant use in the \int f * log f for Normal distribution
const F_LOG_K: f64 = -1.41893853320467274178032973640561763986139747_f64;

impl<B> ModelTrait<B> for NormalModel<B>
where B: AutodiffBackend
{
    fn pdf(&self, data: &Tensor<B, 1>) -> Tensor<B, 1> {
        let v = (self.loc.val() - data.clone()) / (self.scale.val() * 2.0_f64);
        let v = v.powi_scalar(2) * (-1.0_f64);
        let factor = self.scale.val() * SQRT_TWO_PI;
        v.exp().div(factor)
    }

    fn f_log(&self) -> Tensor<B, 1> {
        // calculates 1/2 (-1 - Log[2 \[Pi] s^2]) => -(1/2) - 1/2 Log[2 \[Pi]] - Log[s]
        self.scale.val().log() + F_LOG_K
    }
}

struct Options {
    loc: f64,
    scale: f64,
    num: usize,
    seed: Option<u64>,
    split: bool,
}

fn set_options(options: &mut Options) {
    let mut parser = ArgumentParser::new();

    parser.set_description("Minimum Helliger Distance Estimator for Cauchy-distributed 1D sample and model");

    parser.refer(&mut options.loc)
        .add_argument("loc", Store, "Location of Normal distribution to generate a sample (def 0.0)");

    parser.refer(&mut options.scale)
    .add_argument("scale", Store, "Scale of Normal distribution to generate a sample (def 1.0)");

    parser.refer(&mut options.num)
        .add_argument("num", Store, "Number of observations (def 1000)");

    parser.refer(&mut options.seed)
        .add_option(&["-s", "--seed"], StoreOption, "Provide a seed for reproducibility, otherwise random sample");

    parser.refer(&mut options.split)
        .add_option(&["--split"], StoreTrue, "Use split sample variant");

    parser.parse_args_or_exit();
}

/// Generates Normal distributed sample 
fn generate(options: &Options) -> Vec<f64> {
    let mut rng: ChaCha8Rng = match options.seed {
        Some(val) => ChaCha8Rng::seed_from_u64(val),
        None => ChaCha8Rng::from_entropy(),
    };

    // create random vec
    let dist: Normal = Normal::new(options.loc, options.scale).expect("Wrong parameters for Normal distribution");
    let vec = Vec::from_iter((0..options.num).map(|_| dist.sample(&mut rng)));
    vec
}

fn min_median_max(numbers: &Vec<f64>) -> (f64, f64, f64) {

    let mut to_sort = numbers.clone();
    to_sort.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mid = numbers.len() / 2;
    let med = if numbers.len() % 2 == 0 {
        (numbers[mid - 1] + numbers[mid]) / 2.0
    } else {
        numbers[mid]
    };
    (to_sort[0], med, to_sort[numbers.len()-1])
}


fn normal_model<B: Backend>(vec: &Vec<f64>, device: B::Device) -> NormalModel<B> {
    let (v_min, v_med, v_max) = min_median_max(&vec);

    let loc: Tensor<B, 1> = Tensor::from_data([v_med], &device);
    let scale: Tensor<B, 1> = Tensor::from_data([(v_max - v_min) / 4_f64], &device);

    NormalModel {
        loc: Param::from_tensor(loc),
        scale: Param::from_tensor(scale),
    }
}

type AutoBE = Autodiff<NdArray<f64, i64>>;

fn main() {
    let mut options = Options { loc: 0.0, scale: 1.0, num: 1000, seed: None, split: false };
    set_options(&mut options);

    let device: <AutoBE as Backend>::Device = Default::default();
    let vec = generate(&options);
    let model = normal_model::<AutoBE>(&vec, device);

    println!("Starting params");
    println!("Loc: {}", model.loc.val().clone().into_scalar());
    println!("Scale: {}\n", model.scale.val().clone().into_scalar());

    let (iters, model) = run::<AutoBE, NormalModel<AutoBE>>(
        model,
        vec,
        options.split,
        device,
    );

    println!("Final params (iters={})", iters);
    println!("Loc: {}", model.loc.val().clone().into_scalar());
    println!("Scale: {}\n", model.scale.val().clone().into_scalar());
}
