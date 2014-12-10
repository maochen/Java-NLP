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
    public static final String NE_ORG = "ORGANIZATION";
    public static final String NE_LOC = "LOCATION";

    /**
     * The dependency label for passive.
     */
    public static final String DEP_PASS = "pass";
    /**
     * The dependency label for subjects.
     */
    public static final String DEP_SUBJ = "subj";

    /**
     * The dependency label for adjectival complements.
     */
    public static final String DEP_ACOMP = "acomp";
    /**
     * The dependency label for adverbial clause modifiers.
     */
    public static final String DEP_ADVCL = "advcl";
    /**
     * The dependency label for adverbial modifiers.
     */
    public static final String DEP_ADVMOD = "advmod";
    /**
     * The dependency label for agents.
     */
    public static final String DEP_AGENT = "agent";
    /**
     * The dependency label for adjectival modifiers.
     */
    public static final String DEP_AMOD = "amod";
    /**
     * The dependency label for appositional modifiers.
     */
    public static final String DEP_APPOS = "appos";
    /**
     * The dependency label for attributes.
     */
    public static final String DEP_ATTR = "attr";
    /**
     * The dependency label for auxiliary verbs.
     */
    public static final String DEP_AUX = "aux";
    /**
     * The dependency label for passive auxiliary verbs.
     */
    public static final String DEP_AUXPASS = DEP_AUX + DEP_PASS;
    /**
     * The dependency label for coordinating conjunctions.
     */
    public static final String DEP_CC = "cc";
    /**
     * The dependency label for clausal complements.
     */
    public static final String DEP_CCOMP = "ccomp";
    /**
     * The dependency label for complementizers.
     */
    public static final String DEP_COMPLM = "complm";
    /**
     * The dependency label for conjuncts.
     */
    public static final String DEP_CONJ = "conj";

    // Copula
    public static final String DEP_COP = "cop";
    /**
     * The dependency label for clausal subjects.
     */
    public static final String DEP_CSUBJ = "c" + DEP_SUBJ;
    /**
     * The dependency label for clausal passive subjects.
     */
    public static final String DEP_CSUBJPASS = DEP_CSUBJ + DEP_PASS;
    /**
     * The dependency label for unknown dependencies.
     */
    public static final String DEP_DEP = "dep";
    /**
     * The dependency label for determiners.
     */
    public static final String DEP_DET = "det";
    /**
     * The dependency label for direct objects.
     */
    public static final String DEP_DOBJ = "dobj";
    /**
     * The dependency label for expletives.
     */
    public static final String DEP_EXPL = "expl";
    /**
     * The dependency label for modifiers in hyphenation.
     */
    public static final String DEP_HMOD = "hmod";
    /**
     * The dependency label for hyphenation.
     */
    public static final String DEP_HYPH = "hyph";
    /**
     * The dependency label for indirect objects.
     */
    public static final String DEP_IOBJ = "iobj";
    /**
     * The dependency label for interjections.
     */
    public static final String DEP_INTJ = "intj";
    /**
     * The dependency label for markers.
     */
    public static final String DEP_MARK = "mark";
    /**
     * The dependency label for meta modifiers.
     */
    public static final String DEP_META = "meta";
    /**
     * The dependency label for negation modifiers.
     */
    public static final String DEP_NEG = "neg";
    /**
     * The dependency label for non-finite modifiers.
     */
    public static final String DEP_NFMOD = "nfmod";
    /**
     * The dependency label for infinitival modifiers.
     */
    public static final String DEP_INFMOD = "infmod";
    /**
     * The dependency label for noun phrase modifiers.
     */
    public static final String DEP_NMOD = "nmod";
    /**
     * The dependency label for noun compound modifiers.
     */
    public static final String DEP_NN = "nn";
    /**
     * The dependency label for noun phrase as adverbial modifiers.
     */
    public static final String DEP_NPADVMOD = "npadvmod";
    /**
     * The dependency label for nominal subjects.
     */
    public static final String DEP_NSUBJ = "n" + DEP_SUBJ;
    /**
     * The dependency label for nominal passive subjects.
     */
    public static final String DEP_NSUBJPASS = DEP_NSUBJ + DEP_PASS;
    /**
     * The dependency label for numeric modifiers.
     */
    public static final String DEP_NUM = "num";
    /**
     * The dependency label for elements of compound numbers.
     */
    public static final String DEP_NUMBER = "number";
    /**
     * The dependency label for object predicates.
     */
    public static final String DEP_OPRD = "oprd";
    /**
     * The dependency label for parataxis.
     */
    public static final String DEP_PARATAXIS = "parataxis";
    /**
     * The dependency label for participial modifiers.
     */
    public static final String DEP_PARTMOD = "partmod";
    /**
     * The dependency label for modifiers of prepositions.
     */
    public static final String DEP_PMOD = "pmod";
    /**
     * The dependency label for prepositional complements.
     */
    public static final String DEP_PCOMP = "pcomp";
    /**
     * The dependency label for objects of prepositions.
     */
    public static final String DEP_POBJ = "pobj";
    /**
     * The dependency label for possession modifiers.
     */
    public static final String DEP_POSS = "poss";
    /**
     * The dependency label for possessive modifiers.
     */
    public static final String DEP_POSSESSIVE = "possessive";
    /**
     * The dependency label for pre-conjuncts.
     */
    public static final String DEP_PRECONJ = "preconj";
    /**
     * The dependency label for pre-determiners.
     */
    public static final String DEP_PREDET = "predet";
    /**
     * The dependency label for prepositional modifiers.
     */
    public static final String DEP_PREP = "prep";
    /**
     * The dependency label for particles.
     */
    public static final String DEP_PRT = "prt";
    /**
     * The dependency label for punctuation.
     */
    public static final String DEP_PUNCT = "punct";
    /**
     * The dependency label for modifiers of quantifiers.
     */
    public static final String DEP_QMOD = "qmod";
    /**
     * The dependency label for quantifier phrase modifiers.
     */
    public static final String DEP_QUANTMOD = "quantmod";
    /**
     * The dependency label for relative clause modifiers.
     */
    public static final String DEP_RCMOD = "rcmod";
    /**
     * The dependency label for roots.
     */
    public static final String DEP_ROOT = "root";
    // Only from Stanford
    public static final String DEP_VMOD = "vmod";
    /**
     * The dependency label for open clausal modifiers.
     */
    public static final String DEP_XCOMP = "xcomp";
    /**
     * The dependency label for open clausal subjects.
     */
    public static final String DEP_XSUBJ = "x" + DEP_SUBJ;

}
