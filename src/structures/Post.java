/**
 * 
 */
package structures;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import json.JSONException;
import json.JSONObject;

/**
 * @author hongning
 * @version 0.1
 * @category data structure
 * data structure for a Yelp review document
 * You can create some necessary data structure here to store the processed text content, e.g., bag-of-word representation
 */
public class Post implements Comparable<Post> , Serializable {
	private static final long serialVersionUID = -5858917459160571226L;
	//unique review ID from Yelp
	boolean positive;
	public void setPositive(boolean positive){this.positive = positive;}
	public boolean getPositive(){return positive;}

	double bayesProb;
	public void setBayesProb(double b){
		bayesProb = b;}
	public double getBayesProb(){
		return bayesProb;
	}

	String m_ID;		
	public void setID(String ID) {
		m_ID = ID;
	}
	
	public String getID() {
		return m_ID;
	}

	//author's displayed name
	String m_author;	
	public String getAuthor() {
		return m_author;
	}

	public void setAuthor(String author) {
		this.m_author = author;
	}
	
	//author's location
	String m_location;
	public String getLocation() {
		return m_location;
	}

	public void setLocation(String location) {
		this.m_location = location;
	}

	//review text content
	String m_content;
	public String getContent() {
		return m_content;
	}

	public void setContent(String content) {
		if (!content.isEmpty())
			this.m_content = content;
	}
	
	public boolean isEmpty() {
		return m_content==null || m_content.isEmpty();
	}
	//similarity
	double m_similarity;
	public double getSimilarity(){return m_similarity;}
	public void setSimilarity(double s){m_similarity=s;}

	//timestamp of the post
	String m_date;
	public String getDate() {
		return m_date;
	}

	public void setDate(String date) {
		this.m_date = date;
	}
	
	//overall rating to the business in this review
	double m_rating;
	public double getRating() {
		return m_rating;
	}

	public void setRating(double rating) {
		this.m_rating = rating;
	}

	public Post(String ID) {
		m_ID = ID;
	}
	
	String[] m_tokens; // we will store the tokens 
	public String[] getTokens() {
		return m_tokens;
	}
	
	public void setTokens(String[] tokens) {
		m_tokens = tokens;
	}
	
	HashMap<String, Token> m_vector; // suggested sparse structure for storing the vector space representation with N-grams for this document
	public HashMap<String, Token> getVct() {
		return m_vector;
	}
	
	public void setVct(HashMap<String, Token> vct) {
		m_vector = vct;
	}
	
	public double similiarity(Post p) {
		Iterator iter = m_vector.entrySet().iterator();
		HashMap<String,Token> vct = p.getVct();
		double numerator=0,left=0,right=0;
		while (iter.hasNext()){
			Map.Entry entry = (Map.Entry)iter.next();
			String st = (String)entry.getKey();
			Token token = (Token)entry.getValue();
			Token compared = vct.get(st);
			double weight1 = token.getWeight();
			double weight2;
			if(compared!=null)
				weight2= compared.getWeight();
			else
				weight2=0;
			numerator = numerator+ weight2*weight1;
			left = left + weight1*weight1;
			right = right + weight2*weight2;
		}
		if(right==0)
			return 0;
		double answer = numerator / (Math.sqrt(left)*Math.sqrt(right));
		if(answer==1)
			return 0;
		return answer;//compute the cosine similarity between this post and input p based on their vector space representation
	}

	public int compareTo(Post p){
		if(m_similarity>p.getSimilarity())return -1;
		else if(m_similarity==p.getSimilarity())return 0;
		else
			return 1;
	}

	public Post(String content,int i){
		setContent(content);
	}
	public Post(JSONObject json) {
		m_vector = new HashMap<>();
		setSimilarity(0);
		try {
			//set positive or negative
			String score = json.getString("Overall");
			double positive = Double.parseDouble(score);
			if(positive>=4.0)
				setPositive(true);
			else
				setPositive(false);

			m_ID = json.getString("ReviewID");
			setAuthor(json.getString("Author"));
			setDate(json.getString("Date"));
			setContent(json.getString("Content"));
			setRating(json.getDouble("Overall"));
			setLocation(json.getString("Author_Location"));			
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	public JSONObject getJSON() throws JSONException {
		JSONObject json = new JSONObject();
		
		json.put("ReviewID", m_ID);//must contain
		json.put("Author", m_author);//must contain
		json.put("Date", m_date);//must contain
		json.put("Content", m_content);//must contain
		json.put("Overall", m_rating);//must contain
		json.put("Author_Location", m_location);//must contain
		
		return json;
	}
}
