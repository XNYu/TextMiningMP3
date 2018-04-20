/**
 * 
 */
package structures;

import java.io.Serializable;
import java.util.HashSet;

/**
 * @author hongning
 * Suggested structure for constructing N-gram language model and vector space representation
 */
public class Token implements Serializable {

	int m_id; // the numerical ID you assigned to this token/N-gram
	public int getID() {
		return m_id;
	}
	public int positiveCount;
	public int negativeCount;

	public double informationGain;
	public void setIG(double IG){ informationGain = IG;}
	public double getIG(){ return informationGain;}

	public double posProb,negProb;
	public void setPosProb(double p){
		posProb = p;}
	public void setNegProb(double n){
		negProb = n;}

	public double chiSquare;
	public void setChi(double c){
		chiSquare = c;}
	public double getChi(){
		return chiSquare;
	}
	public void setPositive(boolean positive){
		if(positive)
			positiveCount++;
		else
			negativeCount++;
	}
	public void setID(int id) {
		this.m_id = id;
	}

	String m_token; // the actual text content of this token/N-gram
	public String getToken() {
		return m_token;
	}

	public void setToken(String token) {
		this.m_token = token;
	}

	double m_value; // frequency or count of this token/N-gram
	public double getValue() {
		return m_value;
	}
	public void setValue(double value) {
		this.m_value =value;
	}	
	public void increaseValue(){this.m_value+=1;}

	int m_count;
	public int getCount(){return m_count;}
	public void setCount(int count){m_count=count;}

	double m_weight;
	public double getWeight(){return m_weight;}
	public void setWeight(double weight){m_weight=weight;}

	//default constructor
	public Token(String token) {
		m_token = token;
		m_id = -1;
		positiveCount=0;
		negativeCount=0;
		m_value = 0;
		m_weight= 0;
	}

	//default constructor
	public Token(int id, String token) {
		m_token = token;
		m_id = id;
		positiveCount=0;
		negativeCount=0;
		m_value = 0;
		m_weight= 0;
	}
}
