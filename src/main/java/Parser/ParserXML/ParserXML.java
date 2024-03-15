package main.java.Parser.ParserXML;

import java.io.BufferedReader;
import java.io.InputStream;
import java.util.regex.Matcher;
import java.io.InputStreamReader;
import java.util.regex.Pattern;

public class ParserXML {

    public static NodeXML loadXmlFile(String filename) {
        try {
            InputStream inputStream = Class.class.getResourceAsStream(filename);
            BufferedReader rdr = new BufferedReader(new InputStreamReader(inputStream));
            NodeXML result = Load(rdr);
            rdr.close();
            return result;
        } catch (Exception e) {
            System.err.println("Error in loading xml" + filename);
            System.exit(1);
            return null;
        }
    }

    private static NodeXML Load(BufferedReader reader) throws Exception {
        reader.readLine();
        String str = reader.readLine().trim();
        Matcher match = Pattern.compile("<(.+?)>").matcher(str);
        match.find();
        String[] arrayTage = match.group(1).split(" ");
        String nodeName = arrayTage[0].replace("/", "");
        Matcher matcher = Pattern.compile(">(.+?)<").matcher(str);
        NodeXML node = new NodeXML(nodeName);
        NodeXML child = null;
        int n = arrayTage.length;
        for (int i = 1; i < n; i++) {
            if (arrayTage[i].contains("=")) {
                Matcher name = Pattern.compile("(.+?)=").matcher(arrayTage[i]);
                Matcher value = Pattern.compile("\"(.+?)\"").matcher(arrayTage[i]);
                value.find();
                name.find();
                node.parameters.put(name.group(1), value.group(1));
            }
        }
        if (matcher.find()) {
            node.data = matcher.group(1);
        }
        if (Pattern.compile("(</|/>)").matcher(str).find()) {
            return node;
        }
        while (true) {
            child = Load(reader);
            if (child == null) {
                break;
            }
            node.applyChild(child);
        }
        return node;
    }

}
