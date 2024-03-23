`timescale 1ns/1ps

module power_of_8 (
    input clk, reset_n,
    input   [7:0] i_a,
    output  [31:0]o_output
);
    reg [7:0] ff_2;
    reg [31:0] ff_4;
    reg [31:0] ff_8;

    wire [7:0] wire_2;
    wire [31:0] wire_4;
    wire [31:0] wire_8;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin 
            ff_2 <= 8'd0;
            ff_4 <=32'd0;
            ff_8 <= 32'd0;
        end else begin
            ff_2 <= wire_2;
            ff_4 <= wire_4;
            ff_8 <= wire_8;
        end
    end
 
    assign wire_2 = i_a * i_a;
    assign wire_4 = ff_2 * ff_2;
    assign wire_8 = ff_4 * ff_4;
    assign o_output = ff_8;

endmodule