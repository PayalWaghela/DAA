// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Student {

    struct StudentStruct {
        string name;
        uint256 rollno;
    }

    StudentStruct[] public stuarr;

    function addStudent(string memory name, uint256 rollno) public {
        for (uint i = 0; i < stuarr.length; i++) {
            if (stuarr[i].rollno == rollno) {
                revert("Student with this roll number already exists");
            }
        }
        stuarr.push(StudentStruct(name, rollno));
    }

    function getStudentLength() public view returns (uint) {
        return stuarr.length;
    }

    function displayAllStudents() public view returns (StudentStruct[] memory) {
        return stuarr;
    }

    function getStudentByIndex(uint idx) public view returns (StudentStruct memory) {
        require(idx < stuarr.length, "Index out of bounds");
        return stuarr[idx];
    }

    fallback() external payable {
        // Handles calls to non-existent functions
    }

    receive() external payable {
        // Handles plain ether transfers
    }
}
