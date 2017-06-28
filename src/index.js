import Matrix, {SVD} from 'ml-matrix';
import BaseRegression from 'ml-regression-base';

export default class MultivariateLinearRegression extends BaseRegression {
    constructor(x, y, options = {}) {
        const {
            intercept = true
        } = options;
        super();
        if (x === true) {
            this.weights = y.weights;
            this.inputs = y.inputs;
            this.outputs = y.outputs;
            this.intercept = y.intercept;
        } else {
            if (intercept) {
                x = new Matrix(x);
                x.addColumn(new Array(x.length).fill(1));
            }
            this.weights = new SVD(x, {autoTranspose: true}).solve(y).to2DArray();
            this.inputs = x[0].length;
            this.outputs = y[0].length;
            if (intercept) this.inputs--;
            this.intercept = intercept;
        }
    }

    predict(x) {
        if (Array.isArray(x)) {
            if (typeof x[0] === 'number') {
                return this._predict(x);
            } else if (Array.isArray(x[0])) {
                const y = new Array(x.length);
                for (let i = 0; i < x.length; i++) {
                    y[i] = this._predict(x[i]);
                }
                return y;
            }
        }
        throw new TypeError('x must be a matrix or array of numbers');
    }

    _predict(x) {
        const result = new Array(this.outputs);
        if (this.intercept) {
            for (let i = 0; i < this.outputs; i++) {
                result[i] = this.weights[this.inputs][i];
            }
        } else {
            result.fill(0);
        }
        for (let i = 0; i < this.inputs; i++) {
            for (let j = 0; j < this.outputs; j++) {
                result[j] += this.weights[i][j] * x[i];
            }
        }
        return result;
    }

    score() {
        throw new Error('score method is not implemented yet');
    }

    toJSON() {
        return {
            name: 'multivariateLinearRegression',
            weights: this.weights,
            inputs: this.inputs,
            outputs: this.outputs,
            intercept: this.intercept
        };
    }

    static load(model) {
        if (model.name !== 'multivariateLinearRegression') {
            throw new Error('not a MLR model');
        }
        return new MultivariateLinearRegression(true, model);
    }
}
