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
            const beta = new SVD(x, { autoTranspose: true }).solve(y);
            this.weights = beta.to2DArray();
            this.inputs = x[0].length;
            this.outputs = y[0].length;
            if (intercept) this.inputs--;
            this.intercept = intercept;
			
			/* 
			 * Let's add some basic statistics about the beta's to be able to interpret.
			 * source: http://dept.stat.lsa.umich.edu/~kshedden/Courses/Stat401/Notes/401-multreg.pdf
			 * validated against Excel Regression AddIn
			 */
            const fittedValues = x.mmul(beta);
            const my = new Matrix(y);
            const residuals = my.addM(fittedValues.neg());
            const ro = residuals.to2DArray().map(ri => Math.pow(ri[0],2)).reduce((a,b) => a + b) / (y.length - x.columns);
            this.std_error = Math.sqrt(ro);
            this.std_error_matrix = x.transposeView().mmul(x).pseudoInverse().mul(ro);
            this.std_errors = this.std_error_matrix.diagonal().map(d => Math.sqrt(d));
            this.tstats = this.weights.map((d, i) => this.std_errors[i] === 0 ? 0 : d[0] / this.std_errors[i]);
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
            intercept: this.intercept,
            std_error: this.std_error,
            std_errors: this.std_errors,
            tstats: this.tstats
        };
    }

    static load(model) {
        if (model.name !== 'multivariateLinearRegression') {
            throw new Error('not a MLR model');
        }
        return new MultivariateLinearRegression(true, model);
    }
}
