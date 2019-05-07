import {
  max,
  min,
  pip,
  rep,
  matrixDiag,
  matrixTranspose,
  matrixAdd,
  matrixMultiply,
  matrixChol,
  matrixChol2inv,
  matrixSolve,
  variogramGaussian,
  variogramExponential,
  variogramSpherical,
} from './utils';

// Train using gaussian processes with bayesian priors
function train(t: number[], x: number[], y: number[], model: any, sigma2: any, alpha: number) {
  const variogram = {
    t,
    x,
    y,
    nugget: 0.0,
    range: 0.0,
    sill: 0.0,
    A: 1 / 3,
    n: 0,
    model: variogramExponential,
    K: [],
    M: [],
  };

  switch (model) {
    case 'gaussian':
      variogram.model = variogramGaussian;
      break;
    case 'exponential':
      variogram.model = variogramExponential;
      break;
    case 'spherical':
      variogram.model = variogramSpherical;
      break;
    default:
      variogram.model = variogramExponential;
  }

  // Lag distance/semivariance
  let i;
  let j;
  let k;
  let l;
  let n = t.length;
  const distance = Array((n * n - n) / 2);
  for (i = 0, k = 0; i < n; i++) {
    for (j = 0; j < i; j++, k++) {
      distance[k] = Array(2);
      distance[k][0] = Math.pow(
        Math.pow(x[i] - x[j], 2)
        + Math.pow(y[i] - y[j], 2), 0.5,
      );
      distance[k][1] = Math.abs(t[i] - t[j]);
    }
  }
  distance.sort((a, b) => a[0] - b[0]);
  variogram.range = distance[(n * n - n) / 2 - 1][0];

  // Bin lag distance
  const lags = ((n * n - n) / 2) > 30 ? 30 : (n * n - n) / 2;
  const tolerance = variogram.range / lags;
  const lag = rep(0, lags);
  const semi = rep(0, lags);
  if (lags < 30) {
    for (l = 0; l < lags; l++) {
      lag[l] = distance[l][0];
      semi[l] = distance[l][1];
    }
  } else {
    for (i = 0, j = 0, k = 0, l = 0; i < lags && j < ((n * n - n) / 2); i++, k = 0) {
      while (distance[j][0] <= ((i + 1) * tolerance)) {
        lag[l] += distance[j][0];
        semi[l] += distance[j][1];
        j++;
        k++;
        if (j >= ((n * n - n) / 2)) break;
      }
      if (k > 0) {
        lag[l] /= k;
        semi[l] /= k;
        l++;
      }
    }
    if (l < 2) return variogram; // Error: Not enough points
  }

  // Feature transformation
  n = l;
  variogram.range = lag[n - 1] - lag[0];
  const X = rep(1, 2 * n);
  const Y = Array(n);
  const { A } = variogram;
  for (i = 0; i < n; i++) {
    // eslint-disable-next-line default-case
    switch (model) {
      case 'gaussian':
        X[i * 2 + 1] = 1.0 - Math.exp(-(1.0 / A) * Math.pow(lag[i] / variogram.range, 2));
        break;
      case 'exponential':
        X[i * 2 + 1] = 1.0 - Math.exp(-(1.0 / A) * lag[i] / variogram.range);
        break;
      case 'spherical':
        X[i * 2 + 1] = 1.5 * (lag[i] / variogram.range)
          - 0.5 * Math.pow(lag[i] / variogram.range, 3);
        break;
    }
    Y[i] = semi[i];
  }

  // Least squares
  const Xt = matrixTranspose(X, n, 2);
  let Z = matrixMultiply(Xt, X, 2, n, 2);
  Z = matrixAdd(Z, matrixDiag(1 / alpha, 2), 2, 2);
  const cloneZ = Z.slice(0);
  if (matrixChol(Z, 2)) {
    matrixChol2inv(Z, 2);
  } else {
    matrixSolve(cloneZ, 2);
    Z = cloneZ;
  }
  const W = matrixMultiply(matrixMultiply(Z, Xt, 2, 2, n), Y, 2, n, 1);

  // Variogram parameters
  variogram.nugget = W[0];
  variogram.sill = W[1] * variogram.range + variogram.nugget;
  variogram.n = x.length;

  // Gram matrix with prior
  n = x.length;
  const K = Array(n * n);
  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) {
      K[i * n + j] = variogram.model(Math.pow(Math.pow(x[i] - x[j], 2)
        + Math.pow(y[i] - y[j], 2), 0.5),
        variogram.nugget,
        variogram.range,
        variogram.sill,
        variogram.A);
      K[j * n + i] = K[i * n + j];
    }
    K[i * n + i] = variogram.model(0, variogram.nugget,
      variogram.range,
      variogram.sill,
      variogram.A);
  }

  // Inverse penalized Gram matrix projected to target vector
  let C = matrixAdd(K, matrixDiag(sigma2, n), n, n);
  const cloneC = C.slice(0);
  if (matrixChol(C, n)) {
    matrixChol2inv(C, n);
  } else {
    matrixSolve(cloneC, n);
    C = cloneC;
  }

  // Copy unprojected inverted matrix as K
  const K1 = C.slice(0);
  const M = matrixMultiply(C, t, n, n, 1);
  // @ts-ignore
  variogram.K = K1;
  // @ts-ignore
  variogram.M = M;
  return variogram;
}

// Model prediction
function predict(
  x: number, y: number,
  variogram: {
    n: number;
    model: (arg0: number, arg1: any, arg2: any, arg3: any, arg4: any) => void;
    x: number[];
    y: number[];
    nugget: any;
    range: any;
    sill: any;
    A: any;
    M: number[];
  },
) {
  let i;
  const k = Array(variogram.n);
  for (i = 0; i < variogram.n; i++) {
    k[i] = variogram.model(Math.pow(Math.pow(x - variogram.x[i], 2)
      + Math.pow(y - variogram.y[i], 2), 0.5),
      variogram.nugget, variogram.range,
      variogram.sill, variogram.A);
  }
  return matrixMultiply(k, variogram.M, 1, variogram.n, 1)[0];
}

function variance(
  x: number, y: number,
  variogram: {
    n: number;
    model: {
      (arg0: number, arg1: any, arg2: any, arg3: any, arg4: any): void;
      (arg0: number, arg1: any, arg2: any, arg3: any, arg4: any): number;
    };
    x: number[];
    y: number[];
    nugget: any;
    range: any;
    sill: any;
    A: any;
    K: number[];
  },
) {
  let i;
  const k = Array(variogram.n);
  for (i = 0; i < variogram.n; i++) {
    k[i] = variogram.model(
      Math.pow(Math.pow(x - variogram.x[i], 2) + Math.pow(y - variogram.y[i], 2), 0.5),
      variogram.nugget,
      variogram.range,
      variogram.sill,
      variogram.A,
    );
  }

  const val: number = matrixMultiply(
    matrixMultiply(k, variogram.K,
    1, variogram.n, variogram.n),
    k, 1, variogram.n, 1)[0];

  // @ts-ignore
  return variogram.model(0, variogram.nugget, variogram.range, variogram.sill, variogram.A) + val;
}

// Gridded matrices or contour paths
function grid(
  polygons: number[][][],
  variogram: {
    t: number[];
    n: number;
    model: (arg0: number, arg1: any, arg2: any, arg3: any, arg4: any) => void;
    x: number[];
    y: number[];
    nugget: any;
    range: any;
    sill: any;
    A: any;
    M: number[];
  },
  width: number,
) {
  let i;
  let j;
  let k;
  const n = polygons.length;
  if (n === 0) return;

  // Boundaries of polygons space
  const xlim = [polygons[0][0][0], polygons[0][0][0]];
  const ylim = [polygons[0][0][1], polygons[0][0][1]];
  for (i = 0; i < n; i++) {
    for (j = 0; j < polygons[i].length; j++) { // Vertices
      if (polygons[i][j][0] < xlim[0]) xlim[0] = polygons[i][j][0];
      if (polygons[i][j][0] > xlim[1]) xlim[1] = polygons[i][j][0];
      if (polygons[i][j][1] < ylim[0]) ylim[0] = polygons[i][j][1];
      if (polygons[i][j][1] > ylim[1]) ylim[1] = polygons[i][j][1];
    }
  }

  // Alloc for O(n^2) space
  let xtarget;
  let ytarget;
  const a = Array(2);
  const b = Array(2);
  const lxlim = Array(2); // Local dimensions
  const lylim = Array(2); // Local dimensions
  const x = Math.ceil((xlim[1] - xlim[0]) / width);
  const y = Math.ceil((ylim[1] - ylim[0]) / width);
  const A = Array(x + 1);
  for (i = 0; i <= x; i++) A[i] = Array(y + 1);
  for (i = 0; i < n; i++) {
    // Range for polygons[i]
    lxlim[0] = polygons[i][0][0];
    lxlim[1] = lxlim[0];
    lylim[0] = polygons[i][0][1];
    lylim[1] = lylim[0];
    for (j = 1; j < polygons[i].length; j++) { // Vertices
      if (polygons[i][j][0] < lxlim[0]) lxlim[0] = polygons[i][j][0];
      if (polygons[i][j][0] > lxlim[1]) lxlim[1] = polygons[i][j][0];
      if (polygons[i][j][1] < lylim[0]) lylim[0] = polygons[i][j][1];
      if (polygons[i][j][1] > lylim[1]) lylim[1] = polygons[i][j][1];
    }

    // Loop through polygon subspace
    a[0] = Math.floor(((lxlim[0] - ((lxlim[0] - xlim[0]) % width)) - xlim[0]) / width);
    a[1] = Math.ceil(((lxlim[1] - ((lxlim[1] - xlim[1]) % width)) - xlim[0]) / width);
    b[0] = Math.floor(((lylim[0] - ((lylim[0] - ylim[0]) % width)) - ylim[0]) / width);
    b[1] = Math.ceil(((lylim[1] - ((lylim[1] - ylim[1]) % width)) - ylim[0]) / width);
    for (j = a[0]; j <= a[1]; j++) {
      for (k = b[0]; k <= b[1]; k++) {
        xtarget = xlim[0] + j * width;
        ytarget = ylim[0] + k * width;
        if (pip(polygons[i], xtarget, ytarget)) {
          A[j][k] = predict(xtarget, ytarget, variogram);
        }
      }
    }
  }
  return {
    xlim,
    ylim,
    width,
    data: A,
    zlim: [min(variogram.t), max(variogram.t)],
  };
}

// Plotting on the DOM
function plot(
  canvas: HTMLCanvasElement,
  grid: {
    data: [][],
    xlim: number,
    ylim: number,
    width: number,
    zlim: number,
  },
  xlim: number[],
  ylim: number[],
  colors: any[],
) {
  // Clear screen
  const ctx = canvas.getContext('2d');
  const { data, zlim, width } = grid;
  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Starting boundaries
    const range = [xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]];
    let i;
    let j;
    let x;
    let y;
    let z;
    const n = data.length;
    const m = data[0].length;
    // @ts-ignore
    const wx = Math.ceil(width * canvas.width / (xlim[1] - xlim[0]));
    // @ts-ignore
    const wy = Math.ceil(width * canvas.height / (ylim[1] - ylim[0]));
    for (i = 0; i < n; i++) {
      for (j = 0; j < m; j++) {
        if (data[i][j] === undefined) continue;
        x = canvas.width * (i * width + grid.xlim[0] - xlim[0]) / range[0];
        y = canvas.height * (1 - (j * width + grid.ylim[0] - ylim[0]) / range[1]);
        z = (data[i][j] - zlim[0]) / range[2];
        if (z < 0.0) z = 0.0;
        if (z > 1.0) z = 1.0;
        ctx.fillStyle = colors[Math.floor((colors.length - 1) * z)];
        ctx.fillRect(Math.round(x - wx / 2), Math.round(y - wy / 2), wx, wy);
      }
    }
  }
}

export {
  train,
  predict,
  variance,
  grid,
  plot,
};

export default {
  train,
  predict,
  variance,
  grid,
  plot,
};
