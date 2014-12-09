package org.maochen.datastructure;

/**
 * This is derived from Penn POS
 * <p/>
 * http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
 * <p/>
 * Created by Maochen on 12/8/14.
 */
public class LangLib {
    public static final String POS_CC = "CC"; //Coordinating conjunction
    public static final String POS_CD = "CD"; //Cardinal number
    public static final String POS_DT = "DT"; //Determiner
    public static final String POS_EX = "EX"; //Existential there
    public static final String POS_FW = "FW"; //Foreign word
    public static final String POS_IN = "IN"; //Preposition or subordinating conjunction
    public static final String POS_JJ = "JJ"; //Adjective
    public static final String POS_JJR = "JJR"; //Adjective, comparative
    public static final String POS_JJS = "JJS"; //Adjective, superlative
    public static final String POS_LS = "LS"; //List item marker
    public static final String POS_MD = "MD"; //Modal
    public static final String POS_NN = "NN"; //Noun, singular or mass
    public static final String POS_NNS = "NNS"; //Noun, plural
    public static final String POS_NNP = "NNP"; //Proper noun, singular
    public static final String POS_NNPS = "NNPS"; //Proper noun, plural
    public static final String POS_PDT = "PDT"; //Predeterminer
    public static final String POS_POS = "POS"; //Possessive ending
    public static final String POS_PRP = "PRP"; //Personal pronoun
    public static final String POS_PRPS = "PRP$"; //Possessive pronoun
    public static final String POS_RB = "RB"; //Adverb
    public static final String POS_RBR = "RBR"; //Adverb, comparative
    public static final String POS_RBS = "RBS"; //Adverb, superlative
    public static final String POS_RP = "RP"; //Particle
    public static final String POS_SYM = "SYM"; //Symbol
    public static final String POS_TO = "TO"; //to
    public static final String POS_UH = "UH"; //Interjection
    public static final String POS_VB = "VB"; //Verb, base form
    public static final String POS_VBD = "VBD"; //Verb, past tense
    public static final String POS_VBG = "VBG"; //Verb, gerund or present participle
    public static final String POS_VBN = "VBN"; //Verb, past participle
    public static final String POS_VBP = "VBP"; //Verb, non-3rd person singular present
    public static final String POS_VBZ = "VBZ"; //Verb, 3rd person singular present
    public static final String POS_WDT = "WDT"; //Wh-determiner
    public static final String POS_WP = "WP"; //Wh-pronoun
    public static final String POS_WPS = "WP$"; //Possessive wh-pronoun
    public static final String POS_WRB = "WRB"; //Wh-adverb

    // Named entity tags.
    public static final String NE_DATE = "DATE";
    public static final String NE_TIME = "TIME";
    public static final String NE_PERSON = "PERSON";


    /**
     * The dependency label for passive.
     */
    static public final String DEP_PASS = "pass";
    /**
     * The dependency label for subjects.
     */
    static public final String DEP_SUBJ = "subj";

    /**
     * The dependency label for adjectival complements.
     */
    static public final String DEP_ACOMP = "acomp";
    /**
     * The dependency label for adverbial clause modifiers.
     */
    static public final String DEP_ADVCL = "advcl";
    /**
     * The dependency label for adverbial modifiers.
     */
    static public final String DEP_ADVMOD = "advmod";
    /**
     * The dependency label for agents.
     */
    static public final String DEP_AGENT = "agent";
    /**
     * The dependency label for adjectival modifiers.
     */
    static public final String DEP_AMOD = "amod";
    /**
     * The dependency label for appositional modifiers.
     */
    static public final String DEP_APPOS = "appos";
    /**
     * The dependency label for attributes.
     */
    static public final String DEP_ATTR = "attr";
    /**
     * The dependency label for auxiliary verbs.
     */
    static public final String DEP_AUX = "aux";
    /**
     * The dependency label for passive auxiliary verbs.
     */
    static public final String DEP_AUXPASS = DEP_AUX + DEP_PASS;
    /**
     * The dependency label for coordinating conjunctions.
     */
    static public final String DEP_CC = "cc";
    /**
     * The dependency label for clausal complements.
     */
    static public final String DEP_CCOMP = "ccomp";
    /**
     * The dependency label for complementizers.
     */
    static public final String DEP_COMPLM = "complm";
    /**
     * The dependency label for conjuncts.
     */
    static public final String DEP_CONJ = "conj";
    /**
     * The dependency label for clausal subjects.
     */
    static public final String DEP_CSUBJ = "c" + DEP_SUBJ;
    /**
     * The dependency label for clausal passive subjects.
     */
    static public final String DEP_CSUBJPASS = DEP_CSUBJ + DEP_PASS;
    /**
     * The dependency label for unknown dependencies.
     */
    static public final String DEP_DEP = "dep";
    /**
     * The dependency label for determiners.
     */
    static public final String DEP_DET = "det";
    /**
     * The dependency label for direct objects.
     */
    static public final String DEP_DOBJ = "dobj";
    /**
     * The dependency label for expletives.
     */
    static public final String DEP_EXPL = "expl";
    /**
     * The dependency label for modifiers in hyphenation.
     */
    static public final String DEP_HMOD = "hmod";
    /**
     * The dependency label for hyphenation.
     */
    static public final String DEP_HYPH = "hyph";
    /**
     * The dependency label for indirect objects.
     */
    static public final String DEP_IOBJ = "iobj";
    /**
     * The dependency label for interjections.
     */
    static public final String DEP_INTJ = "intj";
    /**
     * The dependency label for markers.
     */
    static public final String DEP_MARK = "mark";
    /**
     * The dependency label for meta modifiers.
     */
    static public final String DEP_META = "meta";
    /**
     * The dependency label for negation modifiers.
     */
    static public final String DEP_NEG = "neg";
    /**
     * The dependency label for non-finite modifiers.
     */
    static public final String DEP_NFMOD = "nfmod";
    /**
     * The dependency label for infinitival modifiers.
     */
    static public final String DEP_INFMOD = "infmod";
    /**
     * The dependency label for noun phrase modifiers.
     */
    static public final String DEP_NMOD = "nmod";
    /**
     * The dependency label for noun compound modifiers.
     */
    static public final String DEP_NN = "nn";
    /**
     * The dependency label for noun phrase as adverbial modifiers.
     */
    static public final String DEP_NPADVMOD = "npadvmod";
    /**
     * The dependency label for nominal subjects.
     */
    static public final String DEP_NSUBJ = "n" + DEP_SUBJ;
    /**
     * The dependency label for nominal passive subjects.
     */
    static public final String DEP_NSUBJPASS = DEP_NSUBJ + DEP_PASS;
    /**
     * The dependency label for numeric modifiers.
     */
    static public final String DEP_NUM = "num";
    /**
     * The dependency label for elements of compound numbers.
     */
    static public final String DEP_NUMBER = "number";
    /**
     * The dependency label for object predicates.
     */
    static public final String DEP_OPRD = "oprd";
    /**
     * The dependency label for parataxis.
     */
    static public final String DEP_PARATAXIS = "parataxis";
    /**
     * The dependency label for participial modifiers.
     */
    static public final String DEP_PARTMOD = "partmod";
    /**
     * The dependency label for modifiers of prepositions.
     */
    static public final String DEP_PMOD = "pmod";
    /**
     * The dependency label for prepositional complements.
     */
    static public final String DEP_PCOMP = "pcomp";
    /**
     * The dependency label for objects of prepositions.
     */
    static public final String DEP_POBJ = "pobj";
    /**
     * The dependency label for possession modifiers.
     */
    static public final String DEP_POSS = "poss";
    /**
     * The dependency label for possessive modifiers.
     */
    static public final String DEP_POSSESSIVE = "possessive";
    /**
     * The dependency label for pre-conjuncts.
     */
    static public final String DEP_PRECONJ = "preconj";
    /**
     * The dependency label for pre-determiners.
     */
    static public final String DEP_PREDET = "predet";
    /**
     * The dependency label for prepositional modifiers.
     */
    static public final String DEP_PREP = "prep";
    /**
     * The dependency label for particles.
     */
    static public final String DEP_PRT = "prt";
    /**
     * The dependency label for punctuation.
     */
    static public final String DEP_PUNCT = "punct";
    /**
     * The dependency label for modifiers of quantifiers.
     */
    static public final String DEP_QMOD = "qmod";
    /**
     * The dependency label for quantifier phrase modifiers.
     */
    static public final String DEP_QUANTMOD = "quantmod";
    /**
     * The dependency label for relative clause modifiers.
     */
    static public final String DEP_RCMOD = "rcmod";
    /**
     * The dependency label for roots.
     */
    static public final String DEP_ROOT = "root";
    /**
     * The dependency label for open clausal modifiers.
     */
    static public final String DEP_XCOMP = "xcomp";
    /**
     * The dependency label for open clausal subjects.
     */
    static public final String DEP_XSUBJ = "x" + DEP_SUBJ;
    /**
     * The secondary dependency label for gapping relations.
     */
    static public final String DEP_GAP = "gap";
    /**
     * The secondary dependency label for referents.
     */
    static public final String DEP_REF = "ref";
    /**
     * The secondary dependency label for right node raising.
     */
    static public final String DEP_RNR = "rnr";

}
