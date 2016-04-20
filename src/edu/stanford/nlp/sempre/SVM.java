package edu.stanford.nlp.sempre;

import fig.basic.ListUtils;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.StopWatchSet;

import java.util.List;
import java.util.Map;

/**
 * Created by bingxu on 19/4/16.
 */
public class SVM {
    public static class Options {
        @Option(gloss = "SVMnormalize")boolean normalize = true;
    }
    public static final double BIAS = 1;
    public static final double REGULARIZED_BIAS = 0;
    public static boolean REGULARIZED_BIAS_set = true;
    public static boolean BIAS_set = true;
    double  lambda;
    double  eta0;
    double  wDivisor;
    double  wBias;
    double  t;
    public double learnRate = 0;

    public SVM(double lambda,double eta0){
        this.eta0 = eta0;
        this.wDivisor=1;
        this.wBias = 0;
        this.t = 0;
        this.lambda = lambda;
    }

    // Train on just one example and update the weight vector

    /// Perform one iteration of the SGD algorithm with specified gains
    public void computeExpectedCounts(Params params,Example ex, List<Derivation> derivations,Map<String,Double> counts,FeatureMatcher updateFeatureMatcher)
    {
        int n = derivations.size();
        if(n==0) return;
        wDivisor = wDivisor / (1-learnRate * lambda);

        if (wDivisor > 1e5) renorm(params);

        for(int i=0;i<n;i++){ //TODO all derivations they are computed by the previous params, so they contribute together for one update?
            Derivation deriv = derivations.get(i);
            double s = deriv.score/wDivisor + wBias;
            //update for regularization term
            double y = ex.targetValue.getCompatibility(deriv.value) == 1 ? 1 : -1; //TODO : correct?
            // update for loss term
            double d = Logloss_dloss(s, y);
            if (d != 0)// TODO: wether this is correct??
                deriv.incrementAllFeatureVector(d*learnRate*wDivisor, counts, updateFeatureMatcher);
            //w.add(x, eta * d * wDivisor);// if d!= w*eta*d*wDivisor is the counts
            // same for the bias

        }

        if(BIAS_set){
            double etab = learnRate * 0.01;
            if(REGULARIZED_BIAS_set) {
                wBias *= (1 - etab * lambda);
            }
            wBias += etab * d;// TODO : what should I do with this d???
        }

    }

    public void updateCounts(Parser parser,Example ex, Map<String, Double> counts) {
        computeExpectedCounts(Parser parser,Example ex, ex.predDerivations, counts);
    }

    public void updateWeights(Map<String, Double> counts) {
        StopWatchSet.begin("Learner.updateWeights");
        LogInfo.begin_track("Updating weights");
        double sum = 0;
        for (double v : counts.values()) sum += v * v;
        LogInfo.logs("L2 norm: %s", Math.sqrt(sum));
        params.update(counts);
        counts.clear();
        LogInfo.end_track();
        StopWatchSet.end();
    }



    double testOne(const SVector &x, double y, double *ploss, double *pnerr);

    void test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");

/// Perform one epoch with fixed eta and return cost

    double evaluateEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta)
    {
        SvmSgd clone(*this); // take a copy of the current state
        assert(imin <= imax);
        for (int i=imin; i<=imax; i++)
            clone.trainOne(xp.at(i), yp.at(i), eta);
        double loss = 0;
        double cost = 0;
        for (int i=imin; i<=imax; i++)
            clone.testOne(xp.at(i), yp.at(i), &loss, 0);
        loss = loss / (imax - imin + 1);
        cost = loss + 0.5 * lambda * clone.wnorm();
        // cout << "Trying eta=" << eta << " yields cost " << cost << endl;
        return cost;
    }

    void determineEta0(int imin, int imax, const xvec_t &xp, const yvec_t &yp)
    {
        const double factor = 2.0;
        double loEta = 1;
        double loCost = evaluateEta(imin, imax, xp, yp, loEta);
        double hiEta = loEta * factor;
        double hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
        if (loCost < hiCost)
            while (loCost < hiCost)
            {
                hiEta = loEta;
                hiCost = loCost;
                loEta = hiEta / factor;
                loCost = evaluateEta(imin, imax, xp, yp, loEta);
            }
        else if (hiCost < loCost)
            while (hiCost < loCost)
            {
                loEta = hiEta;
                loCost = hiCost;
                hiEta = loEta * factor;
                hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
            }
        eta0 = loEta;
        cout << "# Using eta0=" << eta0 << endl;
    }


        // logloss(a,y) = log(1+exp(-a*y))
        static double Logloss_loss(double a, double y)
        {
            double z = a * y;
            if (z > 18)
                return Math.exp(-z);
            if (z < -18)
                return -z;
            return Math.log(1 + Math.exp(-z));
        }
        // -dloss(a,y)/da
    static double Logloss_dloss(double a, double y)
    {
        double z = a * y;
        if (z > 18)
            return y * Math.exp(-z);
        if (z < -18)
            return y;
        return y / (1 + Math.exp(z));
    }



// hingeloss(a,y) = max(0, 1-a*y)
static double HingeLoss_loss(double a, double y)
        {
        double z = a * y;
        if (z > 1)
        return 0;
        return 1 - z;
        }
// -dloss(a,y)/da
static double HingeLoss_dloss(double a, double y)
        {
        double z = a * y;
        if (z > 1)
        return 0;
        return y;
        }


    /// Renormalize the weights
    void renorm(Params params){
        if (wDivisor != 1.0)
        {
            params.scaleWeights(1.0 / wDivisor); // I added this method to Params.java
            wDivisor = 1.0;
        }
    }

/// Compute the norm of the weights
    double wnorm(Params params)
    {
        double norm = ListUtils.dot(params.getWeight(),params.getWeights()) / wDivisor / wDivisor;
        #if REGULARIZED_BIAS
        if(REGULARIZED_BIAS_set){
            norm += wBias * wBias
        }
        return norm;
    }


}
