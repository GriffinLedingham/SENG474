public class LogisticReg {
	Double [][] data;
	Double [][] x;
	int N, M;
	Double[] y;

	Double[] w;

	Double eta = 2.0; //called kappa in the lecture

	Double threshold = 0.5; //Default threshold for classification

	public void BuildClassifier(Double [][] data) {
		this.data = data;
		this.N=data.length; //number of rows
		this.M=data[0].length-1; //number of attributes (the last attribute is attribute y)

		//Fill in x
		int n;
		this.x = new Double[N][M];
		for(n=0; n<N && data[n]!=null; n++)
			for(int m=0; m<M; m++)
				x[n][m] = data[n][m];
		N=n;

		this.y = new Double[N];
		for(n=0; n<N; n++) y[n] = data[n][M];

		//Now do the computation.
		this.w = ComputeWeightsWithGD();
	}

	public void ClearClassifier() {
		this.w = null;
	}

	public Double Classify(Double[] xn) {
        Double wx = 0.0;
        
		for(int i = 0; i < xn.length; i++)
        {
			wx += w[i] * xn[i];
        }
        
		double p = 1/(1+Math.exp(-1*wx));
        
		if(p >= 0.5)
        {
			return 1.0;
        }
        else
        {
            return -1.0;
        }
	}

	Double[] Gradient(Double[] w) {
        Double[] g = new Double[M];
        
        for(int i = 0;i<M;i++)
        {
            g[i] = 0.0 ;
        }
        
        for(int i = 0;i<N;i++)
        {
            Double wx = 0.0;
            
            for(int j = 0;j<M;j++)
            {
                wx += w[j]*x[i][j];
            }
            
            for(int j = 0;j < M;j++)
            {
                g[j] += (y[i]*x[i][j])/(1+Math.exp(y[i]*wx));
            }
        }
        
        for(int i = 0;i<M;i++)
        {
            g[i] = g[i]/(-N);
        }
        
        return g;
	}

	Double E(Double[] w) {
        Double output = 0.0;
        
        for(int i=0;i<N;i++)
        {
            Double wx = 0.0;
            
            for(int j = 0;j<M;j++)
            {
                wx += w[j]*x[i][j];
            }
            
            output += Math.log(1+Math.exp(-y[i]*wx));
        }
        
        return output/N;
	}


	Double[] ComputeWeightsWithGD() {
        Double E;
        Double[] g = new Double[M];
        
        w = new Double[M];
        
        for(int i = 0;i<w.length;i++)
        {
            w[i] = 0.0;
        }
        
        for(int i = 0;i<50;i++)
        {
            g = Gradient(w);
            E = E(w);
            
            for(int j = 0;j<M;j++)
            {
                w[j]= w[j] - eta*g[j];
            }
        }
        return w;
	}


	public Double[] getW() {
		return w;
	}

	//Shouldn't be used as w's are computed internally, unless for testing an external w vector.
	public void setW(Double[] w) {
		this.w = w;
	}

	public Double getEta() {
		return eta;
	}

	public void setEta(Double eta) {
		this.eta = eta;
	}

	public Double getThreshold() {
		return threshold;
	}

	public void setThreshold(Double threshold) {
		this.threshold = threshold;
	}


	public static void main(String[] args) throws Exception {

		Double [][] data = {
				{1.0,	1.0,	1.0,	1.0},
				{0.9,	1.0,	1.0,	1.0},
				{0.9,	0.875,	1.0,	1.0},
				{0.7,	0.75,	1.0,	-1.0},
				{0.6,	0.875,	1.0,	-1.0},
				{0.6,	0.875,	1.0,	1.0},
				{0.5,	0.75,	1.0,	-1.0},
				{0.5,	0.8125,	1.0,	-1.0},
				{0.5,	1.0,	1.0,	1.0},
				{0.5,	0.875,	1.0,	-1.0},
				{0.5,	0.875,	1.0,	1.0}
		};

		LogisticReg lr = new LogisticReg();

		lr.BuildClassifier(data);

        System.out.print("w: [");
        for(int i = 0;i<lr.w.length;i++)
        {
            if(i==0)
                System.out.print(lr.w[i]);
            else
                System.out.print(","+lr.w[i]);
        }
        System.out.print("]\n");
	}

}
