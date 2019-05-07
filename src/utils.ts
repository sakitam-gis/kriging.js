// https://en.wikipedia.org/wiki/Duff's_device
// https://v8.dev/blog/elements-kinds
/**
 * search array max value
 * @param source
 */
function max(source: number[]): number {
  return Math.max.apply(null, source);
}

/**
 * search array min value
 * @param source
 */
function min(source: number[]): number {
  return Math.min.apply(null, source);
}

/**
 * get mean value from array number
 * @param source
 */
function mean(source: number[]): number {
  let i = 0;
  let sum = 0;
  const length = source.length;
  for (; i < length; i++) sum += source[i];
  return sum / source.length;
}

/**
 * fill array with number
 * @param source
 * @param n
 */
function rep(source: number, n: number): number[] {
  // https://v8.dev/blog/elements-kinds
  const array = [];
  for (let i = 0; i < n; i++) {
    array.push(source);
  }
  return array;
  // return (new Array(n)).fill(source);
}

function pip(source: any[], x: number, y: number): boolean {
  let i = 0;
  let j = source.length - 1;
  let c = false;
  const length = source.length;
  for (; i < length; j = i++) {
    if (((source[i][1] > y) !== (source[j][1] > y))
      && (x < (source[j][0] - source[i][0]) * (y - source[i][1]) / (source[j][1] - source[i][1]) + source[i][0])) {
      c = !c;
    }
  }
  return c;
}

// Matrix algebra
function matrixDiag(c: number, n: number): number[] {
  let i = 0;
  const Z = rep(0, n * n);
  for (; i < n; i++) {
    Z[i * n + i] = c;
  }
  return Z;
}

function matrixTranspose(X: any[], n: number, m: number): any[] {
  let i = 0;
  let j;
  const Z = Array(m * n);
  for (; i < n; i++) {
    j = 0;
    for (; j < m; j++) {
      Z[j * n + i] = X[i * m + j];
    }
  }
  return Z;
}

function matrixScale(X: number[], c: number, n: number, m: number) {
  let i = 0;
  let j;
  for (; i < n; i++) {
    j = 0;
    for (; j < m; j++) {
      X[i * m + j] *= c;
    }
  }
}

function matrixAdd(X: number[], Y: number[], n: number, m: number): number[] {
  let i = 0;
  let j;
  const Z = Array(n * m);
  for (; i < n; i++) {
    j = 0;
    for (; j < m; j++) {
      Z[i * m + j] = X[i * m + j] + Y[i * m + j];
    }
  }
  return Z;
}

// Naive matrix multiplication
function matrixMultiply(X: number[], Y: number[], n: number, m: number, p: number): number[] {
  let i = 0;
  let j;
  let k;
  const Z = Array(n * p);
  for (; i < n; i++) {
    j = 0;
    for (; j < p; j++) {
      Z[i * p + j] = 0;
      k = 0;
      for (; k < m; k++) {
        Z[i * p + j] += X[i * m + k] * Y[k * p + j];
      }
    }
  }
  return Z;
}

// Cholesky decomposition
function matrixChol(X: number[], n: number): boolean {
  let i;
  let j;
  let k;
  const p = Array(n);
  for (i = 0; i < n; i++) p[i] = X[i * n + i];
  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) p[i] -= X[i * n + j] * X[i * n + j];
    if (p[i] <= 0) return false;
    p[i] = Math.sqrt(p[i]);
    for (j = i + 1; j < n; j++) {
      for (k = 0; k < i; k++) X[j * n + i] -= X[j * n + k] * X[i * n + k];
      X[j * n + i] /= p[i];
    }
  }
  for (i = 0; i < n; i++) X[i * n + i] = p[i];
  return true;
}

// Inversion of cholesky decomposition
function matrixChol2inv(X: number[], n: number) {
  let i;
  let j;
  let k;
  let sum;
  for (i = 0; i < n; i++) {
    X[i * n + i] = 1 / X[i * n + i];
    for (j = i + 1; j < n; j++) {
      sum = 0;
      for (k = i; k < j; k++) sum -= X[j * n + k] * X[k * n + i];
      X[j * n + i] = sum / X[j * n + j];
    }
  }
  for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) X[i * n + j] = 0;
  for (i = 0; i < n; i++) {
    X[i * n + i] *= X[i * n + i];
    for (k = i + 1; k < n; k++) X[i * n + i] += X[k * n + i] * X[k * n + i];
    for (j = i + 1; j < n; j++) for (k = j; k < n; k++) X[i * n + j] += X[k * n + i] * X[k * n + j];
  }
  for (i = 0; i < n; i++) for (j = 0; j < i; j++) X[i * n + j] = X[j * n + i];
}

// Inversion via gauss-jordan elimination
function matrixSolve(X: number[], n: number) {
  const m = n;
  const b = Array(n * n);
  const indxc = Array(n);
  const indxr = Array(n);
  const ipiv = Array(n);
  let i;
  let icol: number = 0;
  let irow: number = 0;
  let j;
  let k;
  let l;
  let ll;
  let big;
  let dum;
  let pivinv;
  let temp;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (i === j) b[i * n + j] = 1;
      else b[i * n + j] = 0;
    }
  }

  for (j = 0; j < n; j++) ipiv[j] = 0;

  for (i = 0; i < n; i++) {
    big = 0;
    for (j = 0; j < n; j++) {
      if (ipiv[j] !== 1) {
        for (k = 0; k < n; k++) {
          if (ipiv[k] === 0) {
            if (Math.abs(X[j * n + k]) >= big) {
              big = Math.abs(X[j * n + k]);
              irow = j;
              icol = k;
            }
          }
        }
      }
    }

    ++(ipiv[icol]);

    if (irow !== icol) {
      for (l = 0; l < n; l++) {
        temp = X[irow * n + l];
        X[irow * n + l] = X[icol * n + l];
        X[icol * n + l] = temp;
      }
      for (l = 0; l < m; l++) {
        temp = b[irow * n + l];
        b[irow * n + l] = b[icol * n + l];
        b[icol * n + l] = temp;
      }
    }
    indxr[i] = irow;
    indxc[i] = icol;

    if (X[icol * n + icol] === 0) return false; // Singular

    pivinv = 1 / X[icol * n + icol];
    X[icol * n + icol] = 1;
    for (l = 0; l < n; l++) X[icol * n + l] *= pivinv;
    for (l = 0; l < m; l++) b[icol * n + l] *= pivinv;

    for (ll = 0; ll < n; ll++) {
      if (ll !== icol) {
        dum = X[ll * n + icol];
        X[ll * n + icol] = 0;
        for (l = 0; l < n; l++) X[ll * n + l] -= X[icol * n + l] * dum;
        for (l = 0; l < m; l++) b[ll * n + l] -= b[icol * n + l] * dum;
      }
    }
  }
  for (l = (n - 1); l >= 0; l--) {
    if (indxr[l] !== indxc[l]) {
      for (k = 0; k < n; k++) {
        temp = X[k * n + indxr[l]];
        X[k * n + indxr[l]] = X[k * n + indxc[l]];
        X[k * n + indxc[l]] = temp;
      }
    }
  }

  return true;
}

// Variogram models
function variogramGaussian(
  h: number, nugget: number,
  range: number, sill: number,
  A: number,
) {
  return nugget + ((sill - nugget) / range)
    * (1.0 - Math.exp(-(1.0 / A) * Math.pow(h / range, 2)));
}

function variogramExponential(
  h: number, nugget: number,
  range: number, sill: number,
  A: number,
) {
  return nugget + ((sill - nugget) / range)
    * (1.0 - Math.exp(-(1.0 / A) * (h / range)));
}

function variogramSpherical(
  h: number, nugget: number,
  range: number, sill: number,
) {
  if (h > range) return nugget + (sill - nugget) / range;
  return nugget + ((sill - nugget) / range)
    * (1.5 * (h / range) - 0.5 * Math.pow(h / range, 3));
}

export {
  max,
  min,
  pip,
  rep,
  mean,
  matrixDiag,
  matrixTranspose,
  matrixScale,
  matrixAdd,
  matrixMultiply,
  matrixChol,
  matrixChol2inv,
  matrixSolve,
  variogramGaussian,
  variogramExponential,
  variogramSpherical,
};
