import {solve} from 'ml-matrix';
import BaseRegression from 'ml-regression-base';

export default class MultivariateLinearRegression extends BaseRegression {
    constructor(x, y) {
        super();
        if (x === true) {
            this.weights = y.weights;
            this.inputs = y.inputs;
            this.outputs = y.outputs;
        } else {
            this.weights = solve(x, y).to2DArray();
            this.inputs = x[0].length;
            this.outputs = y[0].length;
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
        const result = new Array(this.outputs).fill(0);
        for (var i = 0; i < this.inputs; i++) {
            for (var j = 0; j < this.outputs; j++) {
                result[j] += this.weights[i][j] * x[i];
            }
        }
        return result;
    }

    toJSON() {
        return {
            name: 'multivariateLinearRegression',
            weights: this.weights,
            inputs: this.inputs,
            outputs: this.outputs
        };
    }

    static load(model) {
        if (model.name !== 'multivariateLinearRegression') {
            throw new Error('not a MLR model');
        }
        return new MultivariateLinearRegression(true, model);
    }
}
