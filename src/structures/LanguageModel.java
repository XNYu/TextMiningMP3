/**
 * 
 */
package structures;

import java.util.HashMap;
import java.io.*;
import java.util.Map;
import java.util.Set;

/**
 * @author hongning
 * Suggested structure for constructing N-gram language model
 */
public class LanguageModel implements Serializable {
	private static final long serialVersionUID = -5858917459160571246L;
	public int m_N; // N-gram
	public int m_V; // the vocabulary size
	public int length;// the document length
	public boolean inited=false;
	public boolean positive;
	public int positiveCount,negativeCount;
	public void setPositiveCount(int p){
		positiveCount = p;}
	public void setNegativeCount(int n){
		negativeCount = n;}

	public int tempTotalCount,tempCount;
	public HashMap<String, Token> m_model; // sparse structure for storing the maximum likelihood estimation of LM with the seen N-grams
	public LanguageModel m_reference; // pointer to the reference language model for smoothing purpose
	
	public double m_lambda; // parameter for linear interpolation smoothing
	public double m_delta; // parameter for additive smoothing
	public HashMap<String,Integer> MapOfS;

	public LanguageModel(int N, boolean pos) {
		m_N = N;
		m_model = new HashMap<String, Token>();
		m_lambda=1;
		m_delta=0.1;
		MapOfS = new HashMap<>();
		positive = pos;
	}


	public void setReference(LanguageModel l){
		m_reference=l;
	}

	public double calcMLProb(String token) {
		int totalCount=0;

		Token tt = m_reference.m_model.get(token.split("-")[0]);
		if(tt==null)
			return 0;

		try{totalCount = tt.getCount();}
		catch (Exception e){
			System.out.println(e);
			System.out.println(token);
			System.out.println(token.split("-")[0]);
		}
		Token t = m_model.get(token);
		if(t==null)
			return 0;
		int count = t.getCount();
		return (double) count/totalCount;
	}

	public void calcBayesProb(){
		for(Map.Entry e:m_model.entrySet()){
			Token t = (Token) e.getValue();
			String token = (String) e.getKey();
			double count,totalCount;

			if(positive) {
				count = t.positiveCount;
				totalCount = positiveCount;
			}else{
				count = t.negativeCount;
				totalCount = negativeCount;
			}
			double prob = (count+m_lambda) / (totalCount+m_lambda*2);

			if(positive)
				t.setPosProb(prob);
			else
				t.setNegProb(prob);
			m_model.put(token, t);
		}
	}

	public double calcBayesProbForWord(String word){
		if(m_model.containsKey(word))
		{
			if(positive)
				return m_model.get(word).posProb;
			else
				return m_model.get(word).negProb;

		}
		else{
			double totalCount;
			if(positive)
				totalCount = positiveCount;
			else
				totalCount = negativeCount;
			return m_lambda / (totalCount + m_lambda * 2);
		}
	}

	public double calcLinearSmoothedProb(String token) {
		String startToken = token.split("-")[0];
		double answer;

		Token t = m_model.get(token);
		int count;
		if(t==null)
			count=0;
		else
			count = t.getCount();

		if (m_N>1)
		{
			double MLProb = calcMLProb(token);
			answer = (1.0-m_lambda) * MLProb + m_lambda * m_reference.calcLinearSmoothedProb(startToken);
		}
		else
		{answer= (count+0.1)/(length+0.1*m_V);
		}
		return answer;// please use additive smoothing to smooth a unigram language model
	}


	public double calcAbsoluteDiscountedProb(String token){
		String startToken = token.split("-")[0];

		double number;
		Token t = m_model.get(token);
		if(t==null)
			number=0;
		else
			number=t.getCount();

		double lambda;
		int startCount;

		Token tt = m_reference.m_model.get(startToken);
		if(tt!=null)
			startCount=tt.getCount();
		else
			return m_reference.calcLinearSmoothedProb(startToken);

		if(MapOfS.get(startToken)==null)
			lambda=1;
		else
			lambda = MapOfS.get(startToken)*m_delta/startCount;

		double answer = Math.max(number - m_delta,0)/startCount;
		answer += m_reference.calcLinearSmoothedProb(startToken) * lambda;
		return answer;
	}

	public void calculateS(){
		for(String s: m_model.keySet()){
			String start = s.split("-")[0];
			if(!MapOfS.containsKey(start))
			{
				MapOfS.put(start,1);
			}
			else{
				int count = MapOfS.get(start);
				count++;
				MapOfS.put(start,count);
			}
		}
	}

	//We have provided you a simple implementation based on unigram language model, please extend it to bigram (i.e., controlled by m_N)
	public String sampling(String startToken, boolean isLinear) {
		if(m_N==1){
		double prob = Math.random(); // prepare to perform uniform sampling
		for(String token:m_model.keySet()) {
			prob -= calcLinearSmoothedProb(token);
			if (prob<=0)
				return token;
			}
		}
		else if(m_N==2){
			double prob = Math.random();
			if(isLinear){
			for(String token:m_model.keySet()){
				String[] strs = token.split("-");
				String start = strs[0];
				String answer = strs[1];
				if(start.equals(startToken))
					prob -= calcLinearSmoothedProb(token);
				if(prob<=0)
					return answer;
				}
			}
			else{
				for(String token:m_model.keySet()){
					String[] strs = token.split("-");
					String start = strs[0];
					String answer = strs[1];
					if(start.equals(startToken))
						prob -= calcAbsoluteDiscountedProb(token);
					if(prob<=0)
						return answer;
				}
			}
		}
		return null; //How to deal with this special case?
	}

	//We have provided you a simple implementation based on unigram language model, please extend it to bigram (i.e., controlled by m_N)
	public double logLikelihood(Post review,boolean isLinear) {
		double likelihood = 0;
		if(m_N==1){
		for(String token:review.getTokens()) {
			if(token.equals(""))
				continue;
			double ans =calcLinearSmoothedProb(token);
			likelihood += Math.log(ans);
		}}
		else if(m_N==2){
			String[] tokens = review.getTokens();
			if(isLinear)
				for (int i = 0; i < tokens.length-1; i++) {
					if(tokens[i].equals(""))
						continue;
					String token = tokens[i]+"-"+tokens[i+1];
					likelihood += Math.log(calcLinearSmoothedProb(token));
					inited=false;
				}
			else
				for (int i = 0; i < tokens.length-1; i++) {
					if(tokens[i].equals(""))
						continue;
					String token = tokens[i]+"-"+tokens[i+1];
					likelihood += Math.log(calcAbsoluteDiscountedProb(token));
					inited=false;
				}
		}
		return likelihood;
	}
}
