//! Animation sampling and pose evaluation (node transforms, skin joint
//! matrices and morph weights) for a point in time.

use crate::model::{AnimPath, Channel, Interpolation, Model, arr, node_local_matrix};
use glam::{Mat4, Quat, Vec3};
use serde_json::Value;

pub struct Pose {
    pub world: Vec<Mat4>,
    /// Morph weights per node; `None` uses the node/mesh defaults.
    pub weights: Vec<Option<Vec<f32>>>,
}

#[derive(Clone, Copy)]
struct Trs {
    t: Vec3,
    r: Quat,
    s: Vec3,
    /// Nodes given as a matrix keep it unless an animation targets them.
    matrix: Option<Mat4>,
}

/// Evaluates animation `anim` at `time` seconds (looping); `None` is the rest pose.
pub fn evaluate(model: &Model, anim: Option<(usize, f32)>) -> Pose {
    let nodes = arr(&model.json, "nodes");
    let mut trs: Vec<Trs> = nodes.iter().map(base_trs).collect();
    let mut weights = vec![None; nodes.len()];

    if let Some(a) = anim.and_then(|(i, t)| Some((model.animations.get(i)?, t))) {
        let (animation, time) = a;
        let t = if animation.duration > 0.0 { time.rem_euclid(animation.duration) } else { 0.0 };
        for ch in &animation.channels {
            let Some(node) = trs.get_mut(ch.node) else { continue };
            let v = sample(ch, t);
            match ch.path {
                AnimPath::Translation => node.t = Vec3::from_slice(&v),
                AnimPath::Rotation => node.r = Quat::from_slice(&v).normalize(),
                AnimPath::Scale => node.s = Vec3::from_slice(&v),
                AnimPath::Weights => weights[ch.node] = Some(v),
            }
            if ch.path != AnimPath::Weights {
                node.matrix = None;
            }
        }
    }

    let world = model.world_matrices(|i| {
        let n = trs[i];
        n.matrix.unwrap_or_else(|| Mat4::from_scale_rotation_translation(n.s, n.r, n.t))
    });
    Pose { world, weights }
}

fn base_trs(n: &Value) -> Trs {
    let m = node_local_matrix(n);
    let (s, r, t) = m.to_scale_rotation_translation();
    Trs { t, r, s, matrix: n.get("matrix").is_some().then_some(m) }
}

/// Default morph weights for a node: node.weights, then mesh.weights, then zeros.
pub fn default_weights(model: &Model, node: usize, mesh: usize, count: usize) -> Vec<f32> {
    let from = |v: Option<&Value>| -> Option<Vec<f32>> {
        let a = v?.get("weights")?.as_array()?;
        Some((0..count).map(|i| a.get(i).and_then(Value::as_f64).unwrap_or(0.0) as f32).collect())
    };
    from(arr(&model.json, "nodes").get(node))
        .or_else(|| from(arr(&model.json, "meshes").get(mesh)))
        .unwrap_or_else(|| vec![0.0; count])
}

fn sample(ch: &Channel, t: f32) -> Vec<f32> {
    let n = ch.comps;
    let keys = ch.times.len();
    let cubic = ch.interpolation == Interpolation::CubicSpline;
    // Value of key k (the middle element of a cubic spline triple).
    let value = |k: usize| -> &[f32] {
        let base = if cubic { (k * 3 + 1) * n } else { k * n };
        &ch.values[base..base + n]
    };
    if keys == 1 || t <= ch.times[0] {
        return value(0).to_vec();
    }
    if t >= ch.times[keys - 1] {
        return value(keys - 1).to_vec();
    }
    let k = ch.times.partition_point(|&x| x <= t).saturating_sub(1).min(keys - 2);
    let (t0, t1) = (ch.times[k], ch.times[k + 1]);
    let dt = (t1 - t0).max(1e-9);
    let s = ((t - t0) / dt).clamp(0.0, 1.0);

    match ch.interpolation {
        Interpolation::Step => value(k).to_vec(),
        Interpolation::Linear if ch.path == AnimPath::Rotation => {
            let a = Quat::from_slice(value(k));
            let b = Quat::from_slice(value(k + 1));
            a.slerp(b, s).to_array().to_vec()
        }
        Interpolation::Linear => value(k).iter().zip(value(k + 1)).map(|(a, b)| a + (b - a) * s).collect(),
        Interpolation::CubicSpline => {
            let out_tangent = &ch.values[(k * 3 + 2) * n..][..n];
            let in_tangent = &ch.values[((k + 1) * 3) * n..][..n];
            let (p0, p1) = (value(k), value(k + 1));
            let (s2, s3) = (s * s, s * s * s);
            let mut v: Vec<f32> = (0..n)
                .map(|i| {
                    (2.0 * s3 - 3.0 * s2 + 1.0) * p0[i]
                        + (s3 - 2.0 * s2 + s) * dt * out_tangent[i]
                        + (-2.0 * s3 + 3.0 * s2) * p1[i]
                        + (s3 - s2) * dt * in_tangent[i]
                })
                .collect();
            if ch.path == AnimPath::Rotation {
                let q = Quat::from_slice(&v).normalize();
                v = q.to_array().to_vec();
            }
            v
        }
    }
}
