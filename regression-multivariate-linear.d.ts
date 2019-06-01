import { AbstractMatrix, Matrix } from 'ml-matrix';

declare module 'ml-regression-multivariate-linear' {
  export interface MLRModel {
    name: 'multivariateLinearRegression';
  }

  export interface MLROptions {
    intercept?: boolean;
    statistics?: boolean;
  }

  export default class MultivariateLinearRegression {
    stdError: number;
    stdErrorMatrix: Matrix;
    stdErrors: number[];
    tStats: number[];
    
    constructor(
      x: number[][] | AbstractMatrix,
      y: number[][] | AbstractMatrix,
      options?: MLROptions
    );

    static load(model: MLRModel): MultivariateLinearRegression;

    predict(x: number[]): number[];
    predict(x: number[][]): number[][];
    predict(x: AbstractMatrix): Matrix;
    toJSON(): MLRModel;
  }
}
