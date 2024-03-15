package main.java.Parser.ParserXML;

import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
public class NodeXML {

    protected NodeXML(String name) {
        this.name = name;
        this.parameters = new HashMap<String, String>();
        this.children = new HashMap<String, List<NodeXML>>();
    }

    public NodeXML getChildWithAttribute(String childName, String attr, String value) {
        List<NodeXML> children = getChildren(childName);
        if (children == null || children.isEmpty()) {
            return null;
        }
        for (NodeXML child : children) {
            String val = child.parameters.get(attr);
            if (value.equals(val)) {
                return child;
            }
        }
        return null;
    }

    protected void applyChild(NodeXML node) {
        List<NodeXML> list = children.get(node.name);
        if (list == null) {
            list = new ArrayList<NodeXML>();
            children.put(node.name, list);
            list.add(node);
        } else {
            list.add(node);
        }
    }
    public NodeXML getChild(String childName) {
        if (children != null) {
            List<NodeXML> nodes = children.get(childName);
            if (nodes != null && !nodes.isEmpty()) {
                return nodes.get(0);
            }
        }
        return null;

    }

    public List<NodeXML> getChildren(String name) {
        if (children != null) {
            List<NodeXML> children = this.children.get(name);
            if (children != null) {
                return children;
            }
        }
        return new ArrayList<NodeXML>();
    }

    public String data;
    public Map<String, String> parameters;
    public String name;
    private Map<String, List<NodeXML>> children;
}
