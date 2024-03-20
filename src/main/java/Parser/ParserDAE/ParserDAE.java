
package main.java.Parser.ParserDAE;

import main.java.Parser.ParserXML.NodeXML;
import main.java.Parser.ParserXML.ParserXML;

public class ParserDAE {

    public static AnimatedModelData loadColladaModel(String colladaFile, int maxWeights) {
        NodeXML node = ParserXML.loadXmlFile(colladaFile);

        SkinLoader skinLoader = new SkinLoader(node.getChild("library_controllers"), maxWeights);
        SkinningData skinningData = skinLoader.extractSkinData();

        SkeletonLoader jointsLoader = new SkeletonLoader(node.getChild("library_visual_scenes"), skinningData.jointOrder);
        SkeletonData jointsData = jointsLoader.extractBoneData();

        GeometryParser g = new GeometryParser(node.getChild("library_geometries"), skinningData.verticesSkinData);
        MeshData meshData = g.extractModelData();

        return new AnimatedModelData(meshData, jointsData);
    }

}
