[StagingDefinition = [Kind = "FastCopy"]]
section Section1;
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "CombineAll_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "Table"]]]}]
shared CombineAll = let
  ソース = Lakehouse.Contents(null),
  ナビゲーション = ソース{[workspaceId = workspace_id]}[Data],
  #"ナビゲーション 1" = ナビゲーション{[lakehouseId = lakehouse_id]}[Data],
  #"ナビゲーション 2" = #"ナビゲーション 1"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"ナビゲーション 3" = #"ナビゲーション 2"{[Name = "raw"]}[Content],
  #"Content の展開" = Table.ExpandTableColumn(#"ナビゲーション 3", "Content", {"Content", "Name", "Extension", "Date accessed", "Date modified", "Date created", "Attributes", "Folder Path"}, {"Content.1", "Name.1", "Extension.1", "Date accessed.1", "Date modified.1", "Date created.1", "Attributes.1", "Folder Path.1"}),
  フィルター処理された非表示のファイル = Table.SelectRows(#"Content の展開", each [Attributes]?[Hidden]? <> true),
  カスタム関数の呼び出し = Table.AddColumn(フィルター処理された非表示のファイル, "ファイルの変換", each ファイルの変換([Content.1])),
  削除された他の列 = Table.SelectColumns(カスタム関数の呼び出し, {"ファイルの変換"}),
  展開されたテーブル列 = Table.ExpandTableColumn(削除された他の列, "ファイルの変換", Table.ColumnNames(ファイルの変換(#"サンプル ファイル"))),
  変更された列タイプ = Table.TransformColumnTypes(展開されたテーブル列, {{"status", type text}, {"data.PriceCategory", type text}, {"data.Type", type text}, {"data.Region", type text}, {"data.MunicipalityCode", Int64.Type}, {"data.Prefecture", type text}, {"data.Municipality", type text}, {"data.DistrictName", type text}, {"data.TradePrice", Int64.Type}, {"data.PricePerUnit", Int64.Type}, {"data.FloorPlan", type text}, {"data.Area", Int64.Type}, {"data.UnitPrice", Int64.Type}, {"data.LandShape", type text}, {"data.Frontage", type number}, {"data.TotalFloorArea", Int64.Type}, {"data.BuildingYear", type date}, {"data.Structure", type text}, {"data.Use", type text}, {"data.Purpose", type text}, {"data.Direction", type text}, {"data.Classification", type text}, {"data.Breadth", type number}, {"data.CityPlanning", type text}, {"data.CoverageRatio", Int64.Type}, {"data.FloorAreaRatio", Int64.Type}, {"data.Period", type text}, {"data.Renovation", type text}, {"data.Remarks", type text}}),
  フィルター処理された行 = Table.SelectRows(変更された列タイプ, each ([data.PriceCategory] = "不動産取引価格情報") and ([data.Type] = "宅地(土地)")),
  削除された列 = Table.RemoveColumns(フィルター処理された行, {"status", "data.FloorPlan", "data.TotalFloorArea", "data.BuildingYear", "data.Structure", "data.Use", "data.Renovation"}),
  #"Inserted Text Before Delimiter" = Table.AddColumn(削除された列, "Text Before Delimiter", each Text.BeforeDelimiter([data.Period], "年"), type text),
  #"Inserted Text Between Delimiters" = Table.AddColumn(#"Inserted Text Before Delimiter", "Text Between Delimiters", each Text.BetweenDelimiters([data.Period], "第", "四"), type text),
  名前が変更された列 = Table.RenameColumns(#"Inserted Text Between Delimiters", {{"data.PriceCategory", "PriceCategory"}, {"data.Type", "Type"}, {"data.Region", "Region"}, {"data.MunicipalityCode", "MunicipalityCode"}, {"data.Prefecture", "Prefecture"}, {"data.Municipality", "Municipality"}, {"data.DistrictName", "DistrictName"}, {"data.TradePrice", "TradePrice"}, {"data.PricePerUnit", "PricePerUnit"}, {"data.Area", "Area"}, {"data.UnitPrice", "UnitPrice"}, {"data.LandShape", "LandShape"}, {"data.Frontage", "Frontage"}, {"data.Purpose", "Purpose"}, {"data.Direction", "Direction"}, {"data.Classification", "Classification"}, {"data.Breadth", "Breadth"}, {"data.CityPlanning", "CityPlanning"}, {"data.CoverageRatio", "CoverageRatio"}, {"data.FloorAreaRatio", "FloorAreaRatio"}, {"data.Remarks", "Remarks"}, {"Text Before Delimiter", "Year"}, {"Text Between Delimiters", "Quarter"}}),
  #"削除された列 1" = Table.RemoveColumns(名前が変更された列, {"data.Period"}),
  #"変更された列タイプ 1" = Table.TransformColumnTypes(#"削除された列 1", {{"Year", Int64.Type}, {"Quarter", Int64.Type}}),
  追加されたインデックス = Table.AddIndexColumn(#"変更された列タイプ 1", "インデックス", 0, 1, Int64.Type),
  #"名前が変更された列 1" = Table.RenameColumns(追加されたインデックス, {{"インデックス", "Id"}}),
  追加されたカスタム = Table.TransformColumnTypes(Table.AddColumn(#"名前が変更された列 1", "Date", each Date.FromText(Text.From([Year]) & "-" & 
    Text.PadStart(Text.From(([Quarter] - 1) * 3 + 1), 2, "0") & "-01")), {{"Date", type date}}),
  並べ替えられた列 = Table.ReorderColumns(追加されたカスタム, {"Id", "Year", "Quarter", "Date", "PriceCategory", "Type", "Region", "MunicipalityCode", "Prefecture", "Municipality", "DistrictName", "TradePrice", "PricePerUnit", "Area", "UnitPrice", "LandShape", "Purpose", "Frontage", "Direction", "Classification", "Breadth", "CityPlanning", "CoverageRatio", "FloorAreaRatio", "Remarks"})
in
  並べ替えられた列;
shared #"サンプル ファイル" = let
  ソース = Lakehouse.Contents(null),
  ナビゲーション = ソース{[workspaceId = workspace_id]}[Data],
  #"ナビゲーション 1" = ナビゲーション{[lakehouseId = lakehouse_id]}[Data],
  #"ナビゲーション 2" = #"ナビゲーション 1"{[Id = "Files", ItemKind = "Folder"]}[Data],
  #"ナビゲーション 3" = #"ナビゲーション 2"{[Name = "raw"]}[Content],
  #"Content の展開" = Table.ExpandTableColumn(#"ナビゲーション 3", "Content", {"Content", "Name", "Extension", "Date accessed", "Date modified", "Date created", "Attributes", "Folder Path"}, {"Content.1", "Name.1", "Extension.1", "Date accessed.1", "Date modified.1", "Date created.1", "Attributes.1", "Folder Path.1"}),
  フィルター処理された非表示のファイル = Table.SelectRows(#"Content の展開", each [Attributes]?[Hidden]? <> true),
  #"ナビゲーション 4" = フィルター処理された非表示のファイル{0}[Content.1]
in
  #"ナビゲーション 4";
shared パラメーター = let
  パラメーター = #"サンプル ファイル" meta [IsParameterQuery = true, IsParameterQueryRequired = true, Type = type binary, BinaryIdentifier = #"サンプル ファイル"]
in
  パラメーター;
shared #"サンプル ファイル の変換" = let
  ソース = Json.Document(パラメーター, 65001),
  テーブルに変換済み = Table.FromRecords({ソース}),
  #"data が展開されました" = Table.ExpandListColumn(テーブルに変換済み, "data"),
  #"data が展開されました1" = Table.ExpandRecordColumn(#"data が展開されました", "data", {"PriceCategory", "Type", "Region", "MunicipalityCode", "Prefecture", "Municipality", "DistrictName", "TradePrice", "PricePerUnit", "FloorPlan", "Area", "UnitPrice", "LandShape", "Frontage", "TotalFloorArea", "BuildingYear", "Structure", "Use", "Purpose", "Direction", "Classification", "Breadth", "CityPlanning", "CoverageRatio", "FloorAreaRatio", "Period", "Renovation", "Remarks"}, {"data.PriceCategory", "data.Type", "data.Region", "data.MunicipalityCode", "data.Prefecture", "data.Municipality", "data.DistrictName", "data.TradePrice", "data.PricePerUnit", "data.FloorPlan", "data.Area", "data.UnitPrice", "data.LandShape", "data.Frontage", "data.TotalFloorArea", "data.BuildingYear", "data.Structure", "data.Use", "data.Purpose", "data.Direction", "data.Classification", "data.Breadth", "data.CityPlanning", "data.CoverageRatio", "data.FloorAreaRatio", "data.Period", "data.Renovation", "data.Remarks"})
in
  #"data が展開されました1";
[FunctionQueryBinding = "{""exemplarFormulaName"":""サンプル ファイル の変換""}"]
shared ファイルの変換 = (パラメーター as binary) => let
  ソース = Json.Document(パラメーター, 65001),
  テーブルに変換済み = Table.FromRecords({ソース}),
  #"data が展開されました" = Table.ExpandListColumn(テーブルに変換済み, "data"),
  #"data が展開されました1" = Table.ExpandRecordColumn(#"data が展開されました", "data", {"PriceCategory", "Type", "Region", "MunicipalityCode", "Prefecture", "Municipality", "DistrictName", "TradePrice", "PricePerUnit", "FloorPlan", "Area", "UnitPrice", "LandShape", "Frontage", "TotalFloorArea", "BuildingYear", "Structure", "Use", "Purpose", "Direction", "Classification", "Breadth", "CityPlanning", "CoverageRatio", "FloorAreaRatio", "Period", "Renovation", "Remarks"}, {"data.PriceCategory", "data.Type", "data.Region", "data.MunicipalityCode", "data.Prefecture", "data.Municipality", "data.DistrictName", "data.TradePrice", "data.PricePerUnit", "data.FloorPlan", "data.Area", "data.UnitPrice", "data.LandShape", "data.Frontage", "data.TotalFloorArea", "data.BuildingYear", "data.Structure", "data.Use", "data.Purpose", "data.Direction", "data.Classification", "data.Breadth", "data.CityPlanning", "data.CoverageRatio", "data.FloorAreaRatio", "data.Period", "data.Renovation", "data.Remarks"})
in
  #"data が展開されました1";
shared workspace_id = "ab22d4ad-cff4-4d54-8621-e718af0e745d" meta [IsParameterQuery = true, IsParameterQueryRequired = false, Type = type text];
shared lakehouse_id = "e0e959fc-42cd-4286-943d-1db19a3f94c2" meta [IsParameterQuery = true, IsParameterQueryRequired = false, Type = type text];
shared CombineAll_DataDestination = let
  Pattern = Lakehouse.Contents([CreateNavigationProperties = false]),
  Navigation_1 = Pattern{[workspaceId = "ab22d4ad-cff4-4d54-8621-e718af0e745d"]}[Data],
  Navigation_2 = Navigation_1{[lakehouseId = "e0e959fc-42cd-4286-943d-1db19a3f94c2"]}[Data],
  TableNavigation = Navigation_2{[Id = "Transactions", ItemKind = "Table"]}?[Data]?
in
  TableNavigation;
