/**
 * 
 */
package analyzer;

import java.io.*;
import java.util.*;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import org.tartarus.snowball.ext.porterStemmer;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.LanguageModel;
import structures.Post;
import structures.Token;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage 
 * NOTE: the code here is only for demonstration purpose, 
 * please revise it accordingly to maximize your implementation's efficiency!
 */
public class DocAnalyzer {
	//N-gram to be created
	int m_N;
	static int tokenID=0;
	//a list of stopwords
	HashSet<String> m_stopwords;
	HashSet<String> m_vocabulary;
	HashMap<String,Double> m_idf;

	//you can store the loaded reviews in this arraylist for further processing
	ArrayList<Post> m_reviews;
	ArrayList<Post> query;
	int docLength = 0;
	int positiveCount,negativeCount,totalCount;
	//you might need something like this to store the counting statistics for validating Zipf's and computing IDF
	HashMap<String, Token> m_stats;
	HashMap<String, Token> m_stats_bigram;
	/*
	Token's TTF
	 */
	HashSet<Token> tokenList;
	HashMap<String,Boolean> tokenDFCalculated;
	//we have also provided a sample implementation of language model in src.structures.LanguageModel
	Tokenizer m_tokenizer;
	
	//this structure is for language modeling
	LanguageModel m_langModel;
	LanguageModel posLM, negLM;
	
	public DocAnalyzer(String tokenModel, int N) throws InvalidFormatException, FileNotFoundException, IOException {
		m_N = N;

		posLM = new LanguageModel(1, true);
		negLM = new LanguageModel(1, false);

		m_reviews = new ArrayList<Post>();
		query = new ArrayList<Post>();

		m_stopwords = new HashSet<>();
		m_stats = new HashMap<>();
		m_stats_bigram = new HashMap<>();
		positiveCount=0;
		negativeCount=0;
		totalCount=0;
		m_vocabulary = new HashSet<>();
		m_idf = new HashMap<>();

        tokenDFCalculated = new HashMap<>();

		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
	}
	
	//sample code for loading a list of stopwords from file
	//you can manually modify the stopword file to include your newly selected words
	public void LoadStopwords(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				//it is very important that you perform the same processing operation to the loaded stopwords
				//otherwise it won't be matched in the text content
				line = SnowballStemming(Normalization(line));
				if (!line.isEmpty()){
					m_stopwords.add(line);
					}
			}
			reader.close();
			System.out.format("Loading %d stopwords from %s\n", m_stopwords.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}

	public void LoadVocabulary(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				//it is very important that you perform the same processing operation to the loaded stopwords
				//otherwise it won't be matched in the text content
//				line = SnowballStemming(Normalization(line));
				if (!line.isEmpty()){
					m_vocabulary.add(line);
//					String[] idfs = line.split("\t");
//					m_vocabulary.add(idfs[0]);
//					double idf = Double.parseDouble(idfs[2]);
//					m_idf.put(idfs[0],idf);
				}
			}
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}

	public boolean hasStopWords(String Ngram){
		if(Ngram.contains("-")){
		String[] tokens = Ngram.split("-");
		for(String t:tokens){
			if(m_stopwords.contains(t))
				return true;
		}
		return false;
		}
		else{
			if(m_stopwords.contains(Ngram))
				return true;
			else
				return false;
		}
	}

	public String[] concat(String[] a, String[] b) {
        String[] c= new String[a.length+b.length];
        System.arraycopy(a, 0, c, 0, a.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
	}  

	public void analyzeDocument(JSONObject json, ArrayList<Post> list) {
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));
				String[] tokens = Tokenize(review.getContent());

				//calculate positive counts
				boolean positive = review.getPositive();


				/**
				 * HINT: essentially you will perform very similar operations as what you have done in analyzeDocument()
				 * Now you should properly update the counts in LanguageModel structure such that we can perform maximum likelihood estimation on it
				 */
				for (int j = 0; j < tokens.length; j++) {
					String token = tokens[j];
					token = Normalization(token);
					if(token.contains("NUM")){
//					increaseNum();
						token = token.replace("NUM","");
					}
					token = SnowballStemming(token);
					tokens[j] = token;
					if(hasStopWords(token))
						tokens[j] = "";
				}
				review.setTokens(tokens);

				/**
				 * HINT: perform necessary text processing here based on the tokenization results
				 * e.g., tokens -> normalization -> stemming -> N-gram -> stopword removal -> to vector
				 * The Post class has defined a "HashMap<String, Token> m_vector" field to hold the vector representation
				 * For efficiency purpose, you can accumulate a term's DF here as well
				*/

//				String[] NGrams = generateNGrams(tokens);
//				String[] TokensWithNGrams = concat(tokens,NGrams);
//
//                for (int j = 0; j < TokensWithNGrams.length; j++) {
//                    String token = TokensWithNGrams[j];
//                    if(hasStopWords(token))
//                        TokensWithNGrams[j]="";
//                }
//
//                review.setTokens(TokensWithNGrams);

                //calculate TF
				HashMap<String,Token> vector = review.getVct();
				for(String token:tokens) {
					if (token.equals(""))
						continue;
					else if (!m_vocabulary.contains(token))
						continue;
					else if (!vector.containsKey(token)) {
						Token t = new Token(1, token);
						t.increaseValue();
						vector.put(token, t);
					}
					else if (vector.containsKey(token)) {
						Token t = vector.get(token);
						t.increaseValue();
						vector.put(token, t);
					}

//calculate DF & positive/negative Counts
                    if(!m_stats.containsKey(token)){
                        Token t = new Token(m_stats.size()+1,token);
                        t.increaseValue();
                        t.setPositive(positive);
                        m_stats.put(token,t);
                        tokenDFCalculated.put(token,true);
                    }
                    else if(m_stats.containsKey(token)&&!tokenDFCalculated.get(token)){
						Token t = m_stats.get(token);
						t.increaseValue();
						t.setPositive(positive);
						m_stats.put(token,t);
						tokenDFCalculated.put(token,true);
					}
				}
				//Calculate Normalized TF and TF*IDF
//				Iterator iter = vector.entrySet().iterator();
//				while (iter.hasNext()){
//					Map.Entry entry = (Map.Entry)iter.next();
//					String st = (String)entry.getKey();
//					Token token = (Token)entry.getValue();
//					double NormTF;
//					double value = token.getValue();
//					if(value>0)
//						NormTF = 1+Math.log10(value);
//					else
//						NormTF = 0;
//					double idf = m_idf.get(st);
//					double weight = NormTF*idf;
//
//					token.setValue(NormTF);
//					token.setWeight(weight);
//					vector.put(st,token);
//				}

				review.setVct(vector);
                clearDFStat();
                if(vector.size()>5)
				{
					list.add(review);
					totalCount++;
					if(positive)
						positiveCount++;
					else
						negativeCount++;
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}

	public void increaseNum(){
		String token = "NUM";
		if(m_stats.containsKey(token)){
			Token t = m_stats.get(token);
			t.increaseValue();
			m_stats.put(token,t);
		}
		else{
			Token t = new Token(m_stats.size()+1,token);
			t.increaseValue();
			m_stats.put(token,t);
		}
	}

	public void createLanguageModel(LanguageModel languageModel) {
		//m_langModel = new LanguageModel(m_N, m_stats.size());

		languageModel.m_model = new HashMap<>(m_stats);
		languageModel.setPositiveCount(positiveCount);
		languageModel.setNegativeCount(negativeCount);
		languageModel.calcBayesProb();

//		for(Post review:m_reviews)
//		{
//
//			String[] tokens = review.getTokens();
//
//			String[] NGrams;
//
//			if(uni==1){
//				for(String token:tokens){
//					docLength++;
//
//                    if(!m_stats.containsKey(token)){
//                        Token t = new Token(m_stats.size()+1,token);
//                        t.increaseValue();
//                        m_stats.put(token,t);
//                    }
//                    else if(m_stats.containsKey(token)){
//						Token t = m_stats.get(token);
//						t.increaseValue();
//						m_stats.put(token,t);
//					}
// 				}
//			}
//
//			else if(uni!=1)
//			{
//				NGrams = generateNGrams(tokens);
//				for(String token:NGrams){
//					if(!token.contains("-"))
//						continue;
//					biLength++;
//					if(!m_stats_bigram.containsKey(token)){
//						Token t = new Token(m_stats_bigram.size()+1,token);
//						t.increaseValue();
//						m_stats_bigram.put(token,t);
//					}
//					else if(m_stats_bigram.containsKey(token)){
//						Token t = m_stats_bigram.get(token);
//						t.increaseValue();
//						m_stats_bigram.put(token,t);
//					}
//				}
//			}
//		}
//
//		if(uni==1)
//		{
//			int vocabularySize = m_stats.size();
//			int length = docLength;
//			HashMap<String,Token> model = m_stats;
//			for(String str:model.keySet()){
//				Token t = model.get(str);
//				double count = t.getValue();
//				double prob = count/length;
//				t.setValue(prob);
//				t.setCount((int) count);
//				model.put(str,t);
//			}
//			languageModel.m_model=model;
//			languageModel.length=docLength;
//		}
//
//		else if(uni!=1)
//		{
//			int length = biLength;
//			HashMap<String,Token> model = m_stats_bigram;
//			for(String str:model.keySet()){
//				Token t = model.get(str);
//				double count = t.getValue();
//				double prob = count/length;
//				t.setValue(prob);
//				t.setCount((int) count);
//				model.put(str,t);
//			}
//			languageModel.m_model=model;
//			languageModel.length = biLength;
//		}

	}
	
	//sample code for loading a json file
	public JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			
			while((line=reader.readLine())!=null) {
				buffer.append(line);
			}
			reader.close();
			
			return new JSONObject(buffer.toString());
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!", filename);
			e.printStackTrace();
			return null;
		} catch (JSONException e) {
			System.err.format("[Error]Failed to parse json file %s!", filename);
			e.printStackTrace();
			return null;
		}
	}
	
	// sample code for demonstrating how to recursively load files in a directory 
	public void LoadDirectory(String folder, String suffix,ArrayList<Post> list) {
		File dir = new File(folder);
		int size = list.size();
		for (File f : dir.listFiles()) {
			System.out.println(f.getName());
			if (f.isFile() && f.getName().endsWith(suffix))
				analyzeDocument(LoadJson(f.getAbsolutePath()),list);
			else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix, list);
		}
		size = list.size() - size;
		System.out.println("Loading " + size + " review documents from " + folder);
	}

	//sample code for demonstrating how to use Snowball stemmer
	public String SnowballStemming(String token) {
		SnowballStemmer stemmer = new englishStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//sample code for demonstrating how to use Porter stemmer
	public String PorterStemming(String token) {
		porterStemmer stemmer = new porterStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//sample code for demonstrating how to perform text normalization
	//you should implement your own normalization procedure here
	public String Normalization(String token) {
		// remove all non-word characters
		// please change this to removing all English punctuation
		token = token.replaceAll("\\W+", "");
		
		// convert to lower case
		token = token.toLowerCase(); 
		
		// add a line to recognize integers and doubles via regular expression
		token = token.replaceAll("^\\d+(\\.+\\d+)?","NUM");
		// and convert the recognized integers and doubles to a special symbol "NUM"
		
		return token;
	}
	
	String[] Tokenize(String text) {
		return m_tokenizer.tokenize(text);
	}
	
	public void TokenizerDemon(String text) {
		System.out.format("Token\tNormalization\tSnonball Stemmer\tPorter Stemmer\n");
		for(String token:m_tokenizer.tokenize(text)){
			String normal = Normalization(token);
			System.out.format("%s\t%s\t%s\t%s\n", token, normal, SnowballStemming(normal), PorterStemming(normal));
		}

	}

	public String[] generateNGrams(String[] tokens){
		String[] NGrams = new String[tokens.length-m_N+1];
		for (int i = 0; i < tokens.length-m_N+1; i++) {
			String NToken = tokens[i];
            if(NToken.equals(""))
            {
                NGrams[i] = "";
                continue;
            }
			for (int j = i+1; j < i+m_N; j++) {
                while(j<tokens.length&&tokens[j].equals("")){
                    j++;
                }
                if(j==tokens.length)
                    continue;
			    NToken = NToken + "-" + tokens[j];
            }
			NGrams[i] = NToken;
		}
		return NGrams;
	}

    //After each review, clear DF Stat
	public void clearDFStat(){
        Iterator iter = tokenDFCalculated.entrySet().iterator();
        while (iter.hasNext()){
            Map.Entry entry = (Map.Entry)iter.next();
            String t = (String)entry.getKey();
            tokenDFCalculated.put(t,false);
        }

	}

	public double calculateLog(double number){
		if(number==0)
			return 0;
		else
			return number * Math.log(number);
	}
	public void calculateInformationGainAndChi(){
		double p1,p2,p3,p4,p5,p6;
		double pPositive = (double) positiveCount / totalCount;
		double pNegative = (double) negativeCount / totalCount;
		p1 = -calculateLog(pPositive);
		p2 = -calculateLog(pNegative);

		for(Map.Entry e:m_stats.entrySet()){
			Token t = (Token) e.getValue();
			double DF = t.getValue();
			double pToken = DF / totalCount;
			double pPosWhenToken = t.positiveCount / DF;
			p3 = calculateLog(pPosWhenToken);
			double pNegWhenToken = t.negativeCount / DF;
			p4 = calculateLog(pNegWhenToken);

			double pWoToken = 1 - pToken;
			double NoTokenReviewCount = totalCount - DF;
			double NoTokenPos = positiveCount - t.positiveCount;
			double NoTokenNeg = negativeCount - t.negativeCount;
			double pPosNoToken = NoTokenPos / NoTokenReviewCount;
			p5 = calculateLog(pPosNoToken);
			double pNegNoToken = NoTokenNeg / NoTokenReviewCount;
			p6 = calculateLog(pNegNoToken);

			double IG = p1 + p2 + pToken * (p3 + p4) + pWoToken * (p5 + p6);
			t.setIG(IG);

			//Calculate Chi Square
			double TokenPos = t.positiveCount;
			double TokenNeg = t.negativeCount;

			double A = TokenPos;
			double B = NoTokenPos;
			double C = TokenNeg;
			double D = NoTokenNeg;

			double up1 = A + B + C + D;
			double up2 = Math.pow(A * D - B * C, 2);
			double down = (A + C) * (B + D) * (A + B) * (C + D);
			double chi = up1 * up2 / down;
			t.setChi(chi);

			m_stats.put((String) e.getKey(), t);
		}
	}

	public void printIGandChi() throws Exception{
		FileOutputStream fs = new FileOutputStream(new File("InformationGain&ChiSquare.txt"));
		PrintStream p = new PrintStream(fs);
		Iterator iter = m_stats.entrySet().iterator();
		while (iter.hasNext()){
			Map.Entry entry = (Map.Entry)iter.next();
			Token t = (Token)entry.getValue();
			p.println(t.getToken() + "," + t.getIG() + "," + t.getChi());
		}
		p.close();
	}

	public void printDF() throws Exception{
		FileOutputStream fs = new FileOutputStream(new File("DF.txt"));
		PrintStream p = new PrintStream(fs);
		Iterator iter = m_stats.entrySet().iterator();
		while (iter.hasNext()){
			Map.Entry entry = (Map.Entry)iter.next();
			Token t = (Token)entry.getValue();
			p.println(t.getToken()+","+t.getValue());
		}
		p.close();
	}

	public void printMap(HashMap<String,Double> map,String name) throws Exception{
		FileOutputStream fs = new FileOutputStream(new File(name));
		PrintStream p = new PrintStream(fs);
		Iterator iter = map.entrySet().iterator();
		while (iter.hasNext()){
			Map.Entry entry = (Map.Entry)iter.next();

			p.println(entry.getKey()+","+entry.getValue());
		}
		p.close();
	}

	public void taskTwoPointOne(){
		createLanguageModel(posLM);
		createLanguageModel(negLM);
//		try {
//			printTaskTwoPointOne();
//		} catch (Exception e) {
//			System.out.println(e);
//		}
	}
	public void printTaskTwoPointOne() throws Exception{
		FileOutputStream fs = new FileOutputStream(new File("BayesLog.txt"));
		PrintStream p = new PrintStream(fs);
		for(Map.Entry e:m_stats.entrySet()){
			String word = (String) e.getKey();
			double posProb = posLM.calcBayesProbForWord(word);
			double negProb = negLM.calcBayesProbForWord(word);
			double logValue = Math.log(posProb/negProb);
			p.println(word+","+logValue);
		}
		p.close();
	}

	public void taskTwoPointTwo(){
		//first calculate all bayes prediction
		int len = m_reviews.size();
		for (int i = 0; i < len; i++) {
			Post p = m_reviews.get(i);
			double bayesProb = calculateBayes(p);
			p.setBayesProb(bayesProb);
			m_reviews.set(i,p);
		}
		Collections.sort(m_reviews);
		//Store real positive
		int[] realPos = new int[len];
		int pos=0;
		for (int i = 0; i < len; i++) {
			Post p = m_reviews.get(i);
			if(p.getPositive())
				pos++;
			realPos[i] = pos;
		}
		try {
			FileOutputStream fs = new FileOutputStream(new File("PrecisionRealllCurve.txt"));
			PrintStream p = new PrintStream(fs);

			for (int i = 0; i < len; i++) {
				int predictedPos = i + 1;
				double TP = realPos[i];
				double FP = predictedPos - realPos[i];
				double TPFN = positiveCount;
				double precision = TP / (TP + FP);
				double recall = TP / TPFN;
				p.println(recall+","+precision);
			}
			p.close();
		} catch (Exception e) {
		}
	}

	public double calculateBayes(Post post){
		double firstPart = Math.log((double) positiveCount/negativeCount);
		double secondPart=0;
		for(String token:post.getTokens()){
			double posProb = posLM.calcBayesProbForWord(token);
			double negProb = negLM.calcBayesProbForWord(token);
			double toAdd = Math.log(posProb)- Math.log(negProb);
			secondPart += toAdd;
		}
		return firstPart + secondPart;
	}

	public void printReviews(int i,List<Post> list) throws Exception{
		FileOutputStream fs = new FileOutputStream(new File("CWC"+i+".txt"));
		PrintStream p = new PrintStream(fs);
		Iterator iter = m_stats.entrySet().iterator();
		for(Post post:list){
			p.println("Query:"+i);
			p.println(post.getSimilarity());
			p.println(post.getAuthor());
			p.println(post.getDate());
			p.println(post.getContent());
		}
		p.close();
	}

	public List<Post> similarWithQuery(Post q){
		ArrayList<Post> answers = new ArrayList<>();
		for(Post review:m_reviews){
			double similarity = q.similiarity(review);
			review.setSimilarity(similarity);
			answers.add(review);
		}
		Collections.sort(answers);
		Collections.reverse(answers);
		return answers.subList(0,3);
	}

	public void saveModel(LanguageModel lm,String path){
		try
		{
			FileOutputStream fileOut =
					new FileOutputStream(path);
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(lm);
			out.close();
			fileOut.close();
			System.out.println("Serialized data is saved in "+path);
		}catch(IOException i)
		{
			i.printStackTrace();
		}
	}

	public LanguageModel readModel(String path){
		LanguageModel lm;
		try
		{
			FileInputStream fileIn = new FileInputStream(path);
			ObjectInputStream in = new ObjectInputStream(fileIn);
			lm = (LanguageModel) in.readObject();
			in.close();
			fileIn.close();
			System.out.println("read object success");
			return lm;
		}catch(IOException i)
		{
			i.printStackTrace();
			return null;
		}catch(ClassNotFoundException c)
		{
			System.out.println("Class not found");
			c.printStackTrace();
			return null;
		}
	}

	public double calculatePerplexity(Post p,LanguageModel lm,boolean isLinear){
		int m_N = lm.m_N;
		double answer=1;
		String[] tokens = p.getTokens();

		if(m_N==1)
			for(String token:p.getTokens()){
				answer = answer*lm.calcLinearSmoothedProb(token);
			}
		else if(m_N==2)
			if(isLinear)
				for (int i = 0; i < tokens.length-1; i++) {
					if(tokens[i].equals(""))
						continue;
					String token = tokens[i]+"-"+tokens[i+1];
					answer = answer*lm.calcLinearSmoothedProb(token);
				}
			else
				for (int i = 0; i < tokens.length-1; i++) {
					if(tokens[i].equals(""))
						continue;
					String token = tokens[i]+"-"+tokens[i+1];
					answer = answer*lm.calcAbsoluteDiscountedProb(token);
				}
		answer = 1/answer;
		answer = Math.pow(answer,1.0/docLength);
		return answer;
	}

	public double calculatePerplexity2(Post p,LanguageModel lm,boolean isLinear){
		int m_N = lm.m_N;
		double answer=0;
		String[] tokens = p.getTokens();

		if(m_N==1){
			answer = answer + lm.logLikelihood(p,true);
		}
		else if(m_N==2)
			if(isLinear)
					answer = answer + lm.logLikelihood(p,true);
			else
					answer = answer + lm.logLikelihood(p,false);
		answer = (0-answer)/tokens.length;
		answer = Math.exp(answer);
		return answer;
	}

	public double StandardDiviation(double[] x) {
        int m = x.length;
		double sum=0;
        for(int i=0;i<m;i++){//求和  
            sum+=x[i];  
        }  
        double dAve=sum/m;//求平均值
		System.out.println(dAve);
		double dVar=0;
        for(int i=0;i<m;i++){//求方差  
            dVar+=(x[i]-dAve)*(x[i]-dAve);  
        }  
        return Math.sqrt(dVar/m);     
    }  
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {		
		DocAnalyzer analyzer = new DocAnalyzer("./data/Model/en-token.bin", 2);
		//load stopwords

		analyzer.LoadStopwords("./data/StopWords.txt");
		analyzer.LoadVocabulary("./data/IGCHIVocabulary.txt");

        //entry point to deal with a collection of documents
		analyzer.LoadDirectory("./Data/yelp", ".json", analyzer.m_reviews);
		analyzer.taskTwoPointOne();
		analyzer.taskTwoPointTwo();

		System.out.println(analyzer.m_stats.size());
		System.out.println(analyzer.m_reviews.size());
		System.out.println(analyzer.positiveCount+","+analyzer.negativeCount);
//		try{
//			analyzer.printDF();
//		}catch (Exception e){
//			System.out.println(e);
//		}
//		analyzer.calculateInformationGainAndChi();
//		try{
//			analyzer.printIGandChi();
//		}
//		catch (Exception e){
//			System.out.println(e);
//		}
//		analyzer.LoadDirectory("./Data/yelp/train", ".json", analyzer.m_reviews);

//		LanguageModel unigramLM = new LanguageModel(1,0);
//		LanguageModel bigramLinear = new LanguageModel(2,0);
//		analyzer.createLanguageModel(unigramLM);
//		System.out.println("Created Unigram");
//		analyzer.createLanguageModel(bigramLinear);
//		System.out.println("Created Bigram");
//		bigramLinear.setReference(unigramLM);
//
//		analyzer.saveModel(unigramLM,"unigram.ser");
//		analyzer.saveModel(bigramLinear,"bigramLinear.ser");
//
//		unigramLM.m_V = analyzer.m_stats.size();
//		bigramLinear.m_V = analyzer.m_stats_bigram.size();

////		int i=0;
////		for (String str:bigramLinear.m_model.keySet()){
////			if(i<20){
////			System.out.println(bigramLinear.m_model.get(str).getValue() + " " + bigramLinear.m_model.get(str).getCount() );
////			i++;}
////			else
////				break;
////		}
////		System.out.println(bigramLinear.m_model.get(12));
//		unigramLM = analyzer.readModel("unigram.ser");
//		bigramLinear = analyzer.readModel("bigramLinear.ser");

//		LanguageModel bigramAbsolute = bigramLinear;
//		bigramAbsolute.calculateS();
//		analyzer.docLength = unigramLM.length;
//		double[] p1= new double[analyzer.query.size()];
//		double[] p2= new double[analyzer.query.size()];
//		double[] p3= new double[analyzer.query.size()];
//		double plex1=0,plex2=0,plex3=0;
//		double pp1=0,pp2=0,pp3=0;
//		int m =0;
//		for(Post p:analyzer.query)
//		{
//			pp1 = analyzer.calculatePerplexity2(p,unigramLM,true);
//			plex1 += pp1;
//			p1[m] = pp1;
//			pp2 = analyzer.calculatePerplexity2(p,bigramLinear,true);
//			plex2 += pp2;
//			p2[m] = pp2;
//			pp3 = analyzer.calculatePerplexity2(p,bigramAbsolute,false);
//			plex3 += pp3;
//			p3[m] = pp3;
//			m++;
//		}
//		System.out.println("Evaluation Unigram:"+plex1/analyzer.query.size()+"----STD:"+analyzer.StandardDiviation(p1));
//		System.out.println("Evaluation BigramLinear:"+plex2/analyzer.query.size()+"----STD:"+analyzer.StandardDiviation(p2));
//		System.out.println("Evaluation BigramAbsolute:"+plex3/analyzer.query.size()+"----STD:"+analyzer.StandardDiviation(p3));

//MP1 Part2, Generate Sentences
//		HashMap<String,Double> m = new HashMap<>();
//		for (int i = 0; i <10; i++) {
//			String sentence = unigramLM.sampling("",true);
//			while(sentence==null)
//				sentence = unigramLM.sampling("",true);
//			String startToken = sentence;
//			System.out.println(sentence);
//			for (int j = 0; j < 15; j++) {
//				String addToken = bigramLinear.sampling(startToken,true);
//				while(addToken==null)
//					addToken = unigramLM.sampling("",true);
//				startToken = addToken;
//				sentence += " "+addToken;
////				System.out.println(sentence);
//			}
//			System.out.println(sentence);
//			Post p = new Post(sentence,1);
//			p.setTokens(analyzer.Tokenize(sentence));
//			double likelihood = bigramLinear.logLikelihood(p,true);
//			m.put(sentence,likelihood);
//		}
//		try{
//		analyzer.printMap(m,"LinearSentences2.txt");}
//		catch (Exception e){
//			System.out.println(e);
//		}
//		m=new HashMap<>();
//		for (int i = 0; i <10; i++) {
//			String sentence = unigramLM.sampling("",true);
//			String startToken = sentence;
//			for (int j = 0; j < 15; j++) {
//				String addToken = bigramAbsolute.sampling(startToken,false);
//				while(addToken==null)
//					addToken = unigramLM.sampling("",true);
//				startToken = addToken;
//				sentence += " "+addToken;
//			}
//			System.out.println(sentence);
//			Post p = new Post(sentence,1);
//			p.setTokens(analyzer.Tokenize(sentence));
//			double likelihood = bigramAbsolute.logLikelihood(p,false);
//			m.put(sentence,likelihood);
//		}
//		try{
//			analyzer.printMap(m,"AbsoluteSentences2.txt");}
//		catch (Exception e){
//			System.out.println(e);
//		}
//Saving Models


//MP2 Part1
//		HashMap<String,Double> map = new HashMap<>();
//		for(String str:bigramLinear.m_model.keySet())
//		{
//			if(str.split("-")[0].equals("good"))
//				map.put(str,bigramLinear.calcLinearSmoothedProb(str));
//		}
//		try{analyzer.printMap(map,"goodsForLineartest.txt");}catch (Exception e){
//			System.out.println(e);
//		}
//		System.out.println("FInished Linear");
//
//		map = new HashMap<>();
//		for(String str:bigramAbsolute.m_model.keySet())
//		{
//			if(str.split("-")[0].equals("good"))
//				map.put(str,bigramAbsolute.calcAbsoluteDiscountedProb(str));
//		}
//		try{analyzer.printMap(map,"goodsForAtest.txt");}catch (Exception e){
//			System.out.println(e);
//		}

//MP1 Part3
//		analyzer.LoadDirectory("./Data/query", ".json", analyzer.query);
//		int i=0;
//		for(Post q:analyzer.query)
//		{
//			List<Post> answers=new ArrayList<>();
//			answers = analyzer.similarWithQuery(q);
//			try{analyzer.printReviews(i,answers);}
//			catch (Exception e){
//				System.out.println(e);
//			}
//			i++;
//		}

	}

}
